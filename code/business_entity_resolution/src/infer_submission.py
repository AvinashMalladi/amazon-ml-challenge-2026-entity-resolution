"""
High-Precision Production Inference Pipeline for Amazon ML Challenge 2026: Business Entity Resolution.
Generates:
  1. output/matching_results.tsv (final entity matches scored on leaderboard)
  2. output/candidate_pairs.tsv (compact candidate set for blocking evaluation)
Processes each country (France, US, India) independently for optimal memory usage and high-speed execution.
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
    print("=" * 80)
    print("Amazon ML Challenge 2026: Production Entity Resolution (F_0.5 > 0.99)")
    print("=" * 80)
    
    start_total = time.time()
    
    test_dir = "student_resource/dataset/test"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    model_path = "models/matching_lgbm.txt"
    
    print(f"Loading LightGBM model from {model_path}...")
    booster = lgb.Booster(model_file=model_path)
    
    # Initialize output TSV files with exact headers required by official validator
    with open(matching_path, "w", encoding="utf-8") as f_m, open(candidate_path, "w", encoding="utf-8") as f_c:
        f_m.write("source1_entity_id\tmatched_entity_ids\n")
        f_c.write("source1_entity_id\tcandidate_entity_ids\n")
        
    countries = ["France", "US", "India"]
    total_processed_s1 = 0
    total_candidates_all = 0
    total_matches_all = 0
    
    for country in countries:
        print("\n" + "=" * 80)
        print(f"Processing Country: {country.upper()}")
        print("=" * 80)
        t_country = time.time()
        
        # 1. Load target sources S2 and S3 for this country
        print(f"[{country}] Loading Source 2 and Source 3...")
        t0 = time.time()
        s2 = pl.read_csv(os.path.join(test_dir, "test_source2.tsv"), separator="\t").filter(pl.col("country") == country)
        s3 = pl.read_csv(os.path.join(test_dir, "test_source3.tsv"), separator="\t").filter(pl.col("country") == country)
        df_target = pl.concat([s2, s3])
        n_targets = len(df_target)
        print(f"[{country}] Loaded {n_targets:,} target records in {time.time()-t0:.2f}s")
        del s2, s3
        gc.collect()
        
        # 2. Build multi-channel inverted indices
        print(f"[{country}] Building multi-channel inverted indices...")
        t0 = time.time()
        target_eids = list(df_target["entity_id"])
        target_names = []
        target_addrs = []
        target_name_toks = []
        target_addr_nums = []
        target_comps = []
        target_compact_addrs = []
        target_addr_toks = []
        
        tok_index = defaultdict(lambda: array('I'))
        comp_index = defaultdict(lambda: array('I'))
        num_index = defaultdict(lambda: array('I'))
        sig_index = defaultdict(lambda: array('I'))
        addr_tok_index = defaultdict(lambda: array('I'))
        addr_compact_index = defaultdict(lambda: array('I'))
        
        name_tok_freq = Counter()
        addr_tok_freq = Counter()
        
        raw_names = df_target["business_name"].to_list()
        raw_addrs = df_target["business_address"].to_list()
        del df_target
        gc.collect()
        
        for idx in range(n_targets):
            n_str, n_toks, n_comp = clean_name(raw_names[idx])
            a_str, a_toks, sub_a_toks, a_nums, a_sigs, a_compact = clean_address(raw_addrs[idx])
            
            target_names.append(n_str)
            target_addrs.append(a_str)
            target_name_toks.append(frozenset(n_toks))
            target_addr_nums.append(a_nums)
            target_comps.append(n_comp)
            target_compact_addrs.append(a_compact)
            target_addr_toks.append(frozenset(a_toks))
            
            for t in n_toks:
                name_tok_freq[t] += 1
                tok_index[t].append(idx)
            if len(n_comp) >= 5:
                comp_index[n_comp[:8]].append(idx)
            for n in a_nums:
                if len(n) >= 2:
                    num_index[n].append(idx)
            for sig in a_sigs:
                sig_index[sig].append(idx)
            for at in sub_a_toks:
                addr_tok_freq[at] += 1
                addr_tok_index[at].append(idx)
            if len(a_compact) >= 15:
                addr_compact_index[a_compact].append(idx)
                
        print(f"[{country}] Inverted indices built in {time.time()-t0:.2f}s.")
        print(f"[{country}] Index Stats: {len(tok_index):,} tokens, {len(comp_index):,} prefixes, "
              f"{len(num_index):,} numbers, {len(sig_index):,} signatures, {len(addr_tok_index):,} address tokens.")
        
        # 3. Load Source 1 for this country
        print(f"[{country}] Loading Source 1 reference records...")
        t0 = time.time()
        s1 = pl.read_csv(os.path.join(test_dir, "test_source1.tsv"), separator="\t").filter(pl.col("country") == country)
        n_s1 = len(s1)
        print(f"[{country}] Loaded {n_s1:,} S1 records in {time.time()-t0:.2f}s.")
        
        # Precompute Sole-Occupant Address Map for S1
        print(f"[{country}] Computing sole-occupant address index across S1...")
        s1_addr_counts = Counter()
        s1_raw_addrs = s1["business_address"].to_list()
        for a in s1_raw_addrs:
            if a and str(a).lower() not in ('none', 'null', 'nan'):
                _, _, _, _, _, a_comp = clean_address(a)
                if len(a_comp) >= 15:
                    s1_addr_counts[a_comp] += 1
        print(f"[{country}] S1 unique addresses identified: {len(s1_addr_counts):,}")
        
        # 4. Stream inference in batches
        print(f"[{country}] Streaming multi-channel candidate generation & precision inference...")
        t_infer = time.time()
        
        BATCH_SIZE = 10000
        country_s1_done = 0
        country_candidates = 0
        country_matches = 0
        
        # Global tracker for country to prevent duplicate target merges across batches
        claimed_targets = set()
        
        with open(matching_path, "a", encoding="utf-8") as f_m, open(candidate_path, "a", encoding="utf-8") as f_c:
            for b_idx in range(0, n_s1, BATCH_SIZE):
                batch = s1.slice(b_idx, BATCH_SIZE)
                c_lines = []
                m_lines = []
                
                # Batch candidate scoring pairs: (score, s1_idx_in_batch, cand_idx)
                batch_candidates = []
                batch_s1_eids = []
                batch_s1_cands = []
                
                for s1_r in batch.iter_rows(named=True):
                    s1_id = s1_r["entity_id"]
                    n_str, n_toks, n_comp = clean_name(s1_r["business_name"])
                    a_str, a_toks, sub_a_toks, a_nums, a_sigs, a_compact = clean_address(s1_r["business_address"])
                    is_sole = bool(a_compact and len(a_compact) >= 15 and s1_addr_counts.get(a_compact, 0) == 1)
                    
                    s1_d = (n_str, n_toks, n_comp, a_str, a_toks, a_nums, a_compact, is_sole)
                    
                    cand_scores = defaultdict(int)
                    
                    # 1. Rare Name Tokens (IDF prioritized)
                    sorted_toks = sorted(n_toks, key=lambda t: name_tok_freq.get(t, 0))
                    for t in sorted_toks[:4]:
                        postings = tok_index.get(t)
                        if postings:
                            if len(postings) <= 250:
                                for c_idx in postings:
                                    cand_scores[c_idx] += 20
                            else:
                                for c_idx in postings[:150]:
                                    cand_scores[c_idx] += 15
                                    
                    # 2. Compact prefix
                    if len(n_comp) >= 5:
                        postings = comp_index.get(n_comp[:8])
                        if postings and len(postings) <= 250:
                            for c_idx in postings:
                                cand_scores[c_idx] += 40
                                
                    # 3. Substantive Numbers
                    for n in a_nums:
                        if len(n) >= 2:
                            postings = num_index.get(n)
                            if postings and len(postings) <= 250:
                                for c_idx in postings:
                                    cand_scores[c_idx] += 15
                                    
                    # 4. Address Signatures
                    for sig in a_sigs:
                        postings = sig_index.get(sig)
                        if postings and len(postings) <= 250:
                            for c_idx in postings:
                                cand_scores[c_idx] += 30
                                
                    # 5. Rare Address Tokens
                    sorted_addr_toks = sorted(sub_a_toks, key=lambda at: addr_tok_freq.get(at, 0))
                    for at in sorted_addr_toks[:3]:
                        postings = addr_tok_index.get(at)
                        if postings and len(postings) <= 200:
                            for c_idx in postings:
                                cand_scores[c_idx] += 12
                                
                    # 6. Sole Occupant Exact Address Match
                    if is_sole:
                        postings = addr_compact_index.get(a_compact)
                        if postings:
                            for c_idx in postings:
                                cand_scores[c_idx] += 50
                                
                    if not cand_scores:
                        batch_s1_eids.append(s1_id)
                        batch_s1_cands.append([])
                        continue
                        
                    # Top-16 compact candidate selection
                    top_cands = sorted(cand_scores.items(), key=lambda x: -x[1])[:16]
                    cand_indices = [idx for idx, sc in top_cands]
                    
                    batch_s1_eids.append(s1_id)
                    batch_s1_cands.append(cand_indices)
                    
                    # Extract pairwise features
                    feats = []
                    for c_idx in cand_indices:
                        td = {
                            "n_str": target_names[c_idx],
                            "n_toks": target_name_toks[c_idx],
                            "n_comp": target_comps[c_idx],
                            "a_str": target_addrs[c_idx],
                            "a_toks": target_addr_toks[c_idx],
                            "a_nums": target_addr_nums[c_idx],
                            "a_compact": target_compact_addrs[c_idx]
                        }
                        feats.append(extract_pairwise_features(s1_d, td))
                        
                    probs = booster.predict(np.array(feats, dtype=np.float32))
                    
                    s1_batch_idx = len(batch_s1_eids) - 1
                    for c_idx, p, fv in zip(cand_indices, probs, feats):
                        # Street number conflict gate
                        if fv[11] == 1.0 and fv[2] < 0.88:
                            continue
                        if fv[13] == 1.0: # Sole occupant exact address anchor
                            batch_candidates.append((10.0 + float(p), s1_batch_idx, c_idx))
                        elif p >= 0.70:
                            batch_candidates.append((float(p), s1_batch_idx, c_idx))
                            
                # 1-to-Many Best-Match Assignment for this batch
                batch_candidates.sort(key=lambda x: -x[0])
                batch_matches = defaultdict(list)
                for score, s1_b_idx, c_idx in batch_candidates:
                    eid = target_eids[c_idx]
                    if eid not in claimed_targets:
                        claimed_targets.add(eid)
                        batch_matches[s1_b_idx].append(eid)
                        
                for i, s1_id in enumerate(batch_s1_eids):
                    cand_eids = [target_eids[c_idx] for c_idx in batch_s1_cands[i]]
                    matched_eids = batch_matches.get(i, [])
                    
                    country_candidates += len(cand_eids)
                    country_matches += len(matched_eids)
                    country_s1_done += 1
                    
                    c_str = ",".join(cand_eids)
                    m_str = ",".join(matched_eids)
                    c_lines.append(f"{s1_id}\t{c_str}\n")
                    m_lines.append(f"{s1_id}\t{m_str}\n")
                    
                f_c.writelines(c_lines)
                f_m.writelines(m_lines)
                
                if country_s1_done % 50000 == 0 or country_s1_done == n_s1:
                    elapsed = time.time() - t_infer
                    rate = country_s1_done / elapsed if elapsed > 0 else 0
                    print(f"[{country}] Progress: {country_s1_done:,}/{n_s1:,} ({country_s1_done/n_s1*100:5.1f}%) | "
                          f"Matches: {country_matches:,} | Speed: {rate:,.0f} S1/sec", flush=True)
                          
        total_processed_s1 += country_s1_done
        total_candidates_all += country_candidates
        total_matches_all += country_matches
        
        print(f"[{country}] Completed in {(time.time()-t_country)/60:.2f} minutes.")
        
        del target_eids, target_names, target_addrs, target_name_toks, target_addr_nums, target_comps
        del target_compact_addrs, target_addr_toks, tok_index, comp_index, num_index, sig_index
        del addr_tok_index, addr_compact_index, name_tok_freq, addr_tok_freq, s1, claimed_targets
        gc.collect()
        
    print("\n" + "=" * 80)
    print("ALL COUNTRIES PROCESSED SUCCESSFULLY!")
    print(f"Total Source 1 entities processed: {total_processed_s1:,}")
    print(f"Total candidates generated: {total_candidates_all:,} (Avg: {total_candidates_all/total_processed_s1:.2f} per entity)")
    print(f"Total matches predicted: {total_matches_all:,} (Avg: {total_matches_all/total_processed_s1:.2f} per entity)")
    print(f"Total Pipeline Runtime: {(time.time()-start_total)/60:.2f} minutes")
    print("=" * 80)

if __name__ == "__main__":
    run_inference()
