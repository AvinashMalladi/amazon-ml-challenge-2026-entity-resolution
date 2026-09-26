"""
Production Inference Pipeline for Amazon ML Challenge 2026: Business Entity Resolution.
Generates:
  1. output/matching_results.tsv (final entity matches scored on leaderboard)
  2. output/candidate_pairs.tsv (compact candidate set for blocking evaluation)
Processes each country (France, US, India) independently for optimal memory usage and speed.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

import polars as pl
from array import array
from collections import defaultdict, Counter
import numpy as np
import lightgbm as lgb
import time
import gc

from normalization import clean_name, clean_address
from features import extract_pairwise_features

def run_inference():
    print("=" * 70)
    print("Starting Amazon ML Challenge 2026 Entity Resolution Inference Pipeline")
    print("=" * 70)
    
    start_total = time.time()
    
    # Paths
    test_dir = "student_resource/dataset/test"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    model_path = "models/matching_lgbm.txt"
    
    print(f"Loading LightGBM model from {model_path}...")
    booster = lgb.Booster(model_file=model_path)
    
    # Initialize output TSV files with headers
    with open(matching_path, "w", encoding="utf-8") as f_m, open(candidate_path, "w", encoding="utf-8") as f_c:
        f_m.write("source1_entity_id\tmatched_entity_ids\n")
        f_c.write("source1_entity_id\tcandidate_entity_ids\n")
        
    countries = ["France", "US", "India"]
    total_processed_s1 = 0
    total_candidates_all = 0
    total_matches_all = 0
    
    for country in countries:
        print("\n" + "=" * 70)
        print(f"Processing Country: {country.upper()}")
        print("=" * 70)
        t_country = time.time()
        
        # 1. Load target sources S2 and S3 for this country
        print(f"[{country}] Loading Source 2 and Source 3...")
        t0 = time.time()
        s2 = pl.read_csv(os.path.join(test_dir, "test_source2.tsv"), separator="\t").filter(pl.col("country") == country)
        s3 = pl.read_csv(os.path.join(test_dir, "test_source3.tsv"), separator="\t").filter(pl.col("country") == country)
        df_target = pl.concat([s2, s3])
        n_targets = len(df_target)
        print(f"[{country}] Loaded {n_targets} target records (S2: {len(s2)}, S3: {len(s3)}) in {time.time()-t0:.2f}s")
        del s2, s3
        gc.collect()
        
        # 2. Build compact array inverted index
        print(f"[{country}] Building compact inverted indices...")
        t0 = time.time()
        target_eids = list(df_target["entity_id"])
        target_names = []
        target_addrs = []
        target_name_toks = []
        target_addr_nums = []
        target_comps = []
        
        name_index = defaultdict(lambda: array('I'))
        sig_index = defaultdict(lambda: array('I'))
        num_index = defaultdict(lambda: array('I'))
        token_freq = Counter()
        
        raw_names = df_target["business_name"].to_list()
        raw_addrs = df_target["business_address"].to_list()
        del df_target
        gc.collect()
        
        for idx in range(n_targets):
            n_str, n_toks, n_comp = clean_name(raw_names[idx])
            a_str, a_toks, a_nums, a_sigs = clean_address(raw_addrs[idx])
            
            target_names.append(n_str)
            target_addrs.append(a_str)
            target_name_toks.append(frozenset(n_toks))
            target_addr_nums.append(a_nums)
            target_comps.append(n_comp)
            
            for t in n_toks:
                token_freq[t] += 1
                name_index[t].append(idx)
            for sig in a_sigs:
                sig_index[sig].append(idx)
            for n in a_nums:
                if len(n) >= 4:
                    num_index[n].append(idx)
                    
        print(f"[{country}] Inverted index constructed in {time.time()-t0:.2f}s.")
        print(f"[{country}] Index size: {len(name_index)} tokens, {len(sig_index)} signatures, {len(num_index)} numbers.")
        
        # 3. Load Source 1 for this country
        print(f"[{country}] Loading Source 1 records...")
        t0 = time.time()
        s1 = pl.read_csv(os.path.join(test_dir, "test_source1.tsv"), separator="\t").filter(pl.col("country") == country)
        n_s1 = len(s1)
        print(f"[{country}] Loaded {n_s1} S1 records in {time.time()-t0:.2f}s.")
        
        # 4. Stream inference in batches
        print(f"[{country}] Generating candidates & running LightGBM inference...")
        t_infer = time.time()
        
        BATCH_SIZE = 5000
        country_s1_done = 0
        country_candidates = 0
        country_matches = 0
        
        # Open output files in append mode
        with open(matching_path, "a", encoding="utf-8") as f_m, open(candidate_path, "a", encoding="utf-8") as f_c:
            for b_idx in range(0, n_s1, BATCH_SIZE):
                batch = s1.slice(b_idx, BATCH_SIZE)
                
                m_lines = []
                c_lines = []
                
                for s1_r in batch.iter_rows(named=True):
                    s1_id = s1_r["entity_id"]
                    n_str, n_toks, n_comp = clean_name(s1_r["business_name"])
                    a_str, a_toks, a_nums, a_sigs = clean_address(s1_r["business_address"])
                    
                    s1_data = {
                        "name_str": n_str, "name_toks": n_toks, "name_comp": n_comp,
                        "addr_str": a_str, "addr_toks": a_toks, "addr_nums": a_nums
                    }
                    
                    # Candidate accumulation
                    cand_scores = defaultdict(int)
                    
                    # Channel 1: Name tokens ranked by frequency
                    sorted_toks = sorted(n_toks, key=lambda t: token_freq[t])
                    for t in sorted_toks[:4]:
                        postings = name_index.get(t)
                        if postings and len(postings) <= 250:
                            for c_idx in postings:
                                cand_scores[c_idx] += 12
                                
                    # Channel 2: Address Signatures (street number + street word)
                    for sig in a_sigs:
                        postings = sig_index.get(sig)
                        if postings and len(postings) <= 250:
                            for c_idx in postings:
                                cand_scores[c_idx] += 30
                                
                    # Channel 3: Rare Numbers (PIN code / phone / unit)
                    for n in a_nums:
                        if len(n) >= 4:
                            postings = num_index.get(n)
                            if postings and len(postings) <= 250:
                                for c_idx in postings:
                                    cand_scores[c_idx] += 15
                                    
                    if not cand_scores:
                        # Singleton: zero candidates, zero matches
                        c_lines.append(f"{s1_id}\t\n")
                        m_lines.append(f"{s1_id}\t\n")
                        country_s1_done += 1
                        continue
                        
                    # Top-12 candidate selection
                    top_cands = sorted(cand_scores.items(), key=lambda x: -x[1])[:12]
                    cand_indices = [idx for idx, sc in top_cands]
                    cand_eids = [target_eids[idx] for idx in cand_indices]
                    
                    country_candidates += len(cand_eids)
                    
                    # Feature Extraction
                    feats = []
                    for c_idx in cand_indices:
                        cand_data = {
                            "name_str": target_names[c_idx],
                            "name_toks": target_name_toks[c_idx],
                            "name_comp": target_comps[c_idx],
                            "addr_str": target_addrs[c_idx],
                            "addr_toks": set(target_addrs[c_idx].split()),
                            "addr_nums": target_addr_nums[c_idx]
                        }
                        feats.append(extract_pairwise_features(s1_data, cand_data))
                        
                    probs = booster.predict(np.array(feats, dtype=np.float32))
                    
                    # Optimal threshold for Macro F_0.5 (high precision)
                    matched_eids = [cand_eids[i] for i, p in enumerate(probs) if p >= 0.72]
                    country_matches += len(matched_eids)
                    
                    # Write lines
                    c_str = ",".join(cand_eids)
                    m_str = ",".join(matched_eids)
                    
                    c_lines.append(f"{s1_id}\t{c_str}\n")
                    m_lines.append(f"{s1_id}\t{m_str}\n")
                    country_s1_done += 1
                    
                f_c.writelines(c_lines)
                f_m.writelines(m_lines)
                
                if country_s1_done % 25000 < BATCH_SIZE or country_s1_done == n_s1:
                    elapsed = time.time() - t_infer
                    rate = country_s1_done / elapsed if elapsed > 0 else 0
                    print(f"[{country}] Progress: {country_s1_done:,}/{n_s1:,} ({country_s1_done/n_s1*100:.1f}%) | "
                          f"Rate: {rate:.0f} ent/s | Avg Cands: {country_candidates/max(1, country_s1_done):.1f} | "
                          f"Avg Matches: {country_matches/max(1, country_s1_done):.2f}")
                          
        total_processed_s1 += country_s1_done
        total_candidates_all += country_candidates
        total_matches_all += country_matches
        
        print(f"[{country}] Completed in {(time.time()-t_country)/60:.2f} minutes.")
        
        # Clean up memory
        del target_eids, target_names, target_addrs, target_name_toks, target_addr_nums, target_comps
        del name_index, sig_index, num_index, token_freq, s1
        gc.collect()
        
    print("\n" + "=" * 70)
    print("ALL COUNTRIES PROCESSED SUCCESSFULLY!")
    print(f"Total Source 1 entities processed: {total_processed_s1:,}")
    print(f"Total candidates generated: {total_candidates_all:,} (Avg: {total_candidates_all/total_processed_s1:.2f} per entity)")
    print(f"Total matches predicted: {total_matches_all:,} (Avg: {total_matches_all/total_processed_s1:.2f} per entity)")
    print(f"Total Pipeline Runtime: {(time.time()-start_total)/60:.2f} minutes")
    print("=" * 70)

if __name__ == "__main__":
    run_inference()
