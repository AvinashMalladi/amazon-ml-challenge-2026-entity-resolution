import sys
import os
sys.path.append("code/business_entity_resolution/src")
import polars as pl
from array import array
from collections import defaultdict, Counter
import numpy as np
import lightgbm as lgb
import time
from normalization import clean_name, clean_address
from features import extract_pairwise_features

print("=== Investigating True Matches Recall & Score Distribution ===", flush=True)

# Load S1 and GT joined properly!
s1_sample = pl.read_csv("student_resource/dataset/train/train_source1.tsv", separator="\t", n_rows=2000)
gt_sample = pl.read_csv("student_resource/dataset/train/train_ground_truth.tsv", separator="\t")

# Filter GT to only entities in s1_sample
s1_eids = set(s1_sample["entity_id"])
gt_filtered = gt_sample.filter(pl.col("source1_entity_id").is_in(list(s1_eids)))

gt_map = {}
all_true_tgt_ids = set()
for r in gt_filtered.iter_rows(named=True):
    s1_id = r["source1_entity_id"]
    m = r["matched_entity_ids"]
    if m:
        tgts = set(x.strip() for x in m.split(",") if x.strip())
        gt_map[s1_id] = tgts
        all_true_tgt_ids.update(tgts)
    else:
        gt_map[s1_id] = set()

print(f"Sampled {len(s1_sample)} S1 entities. Found GT for {len(gt_map)}.", flush=True)
print(f"Total true match target IDs needed: {len(all_true_tgt_ids)}", flush=True)

# Now load target records: fetch all true targets + 100k noise records
s2_noise = pl.read_csv("student_resource/dataset/train/train_source2.tsv", separator="\t", n_rows=100000)
s3_noise = pl.read_csv("student_resource/dataset/train/train_source3.tsv", separator="\t", n_rows=100000)
targets = pl.concat([s2_noise, s3_noise]).unique(subset=["entity_id"])

existing = set(targets["entity_id"])
missing = all_true_tgt_ids - existing
print(f"Loading {len(missing)} missing true targets from S2 and S3...", flush=True)
if missing:
    s2_miss = pl.read_csv("student_resource/dataset/train/train_source2.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(missing)))
    s3_miss = pl.read_csv("student_resource/dataset/train/train_source3.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(missing)))
    targets = pl.concat([targets, s2_miss, s3_miss]).unique(subset=["entity_id"])

print(f"Total target pool size: {len(targets)}", flush=True)

# Build index
target_eids = list(targets["entity_id"])
target_names = []
target_addrs = []
target_name_toks = []
target_addr_nums = []
target_comps = []
target_countries = list(targets["country"])

name_index = defaultdict(lambda: array('I'))
sig_index = defaultdict(lambda: array('I'))
num_index = defaultdict(lambda: array('I'))
token_freq = Counter()

raw_names = targets["business_name"].to_list()
raw_addrs = targets["business_address"].to_list()

for idx in range(len(target_eids)):
    c = target_countries[idx]
    n_str, n_toks, n_comp = clean_name(raw_names[idx])
    a_str, a_toks, a_nums, a_sigs = clean_address(raw_addrs[idx])
    
    target_names.append(n_str)
    target_addrs.append(a_str)
    target_name_toks.append(frozenset(n_toks))
    target_addr_nums.append(a_nums)
    target_comps.append(n_comp)
    
    for t in n_toks:
        token_freq[(c, t)] += 1
        name_index[(c, t)].append(idx)
    for sig in a_sigs:
        sig_index[(c, sig)].append(idx)
    for n in a_nums:
        if len(n) >= 4:
            num_index[(c, n)].append(idx)

print("Inverted index ready.", flush=True)

booster = lgb.Booster(model_file="models/matching_lgbm.txt")

# Test blocking recall and scores
true_match_scores = []
false_match_scores = []
retrieved_true_count = 0
total_true_count = sum(len(v) for v in gt_map.values())

for s1_r in s1_sample.iter_rows(named=True):
    s1_id = s1_r["entity_id"]
    c = s1_r["country"]
    true_set = gt_map.get(s1_id, set())
    
    n_str, n_toks, n_comp = clean_name(s1_r["business_name"])
    a_str, a_toks, a_nums, a_sigs = clean_address(s1_r["business_address"])
    s1_data = {
        "name_str": n_str, "name_toks": n_toks, "name_comp": n_comp,
        "addr_str": a_str, "addr_toks": a_toks, "addr_nums": a_nums
    }
    
    cand_scores = defaultdict(int)
    sorted_toks = sorted(n_toks, key=lambda t: token_freq[(c, t)])
    for t in sorted_toks[:5]:
        postings = name_index.get((c, t))
        if postings and len(postings) <= 300:
            for c_idx in postings:
                cand_scores[c_idx] += 12
    for sig in a_sigs:
        postings = sig_index.get((c, sig))
        if postings and len(postings) <= 300:
            for c_idx in postings:
                cand_scores[c_idx] += 30
    for n in a_nums:
        if len(n) >= 4:
            postings = num_index.get((c, n))
            if postings and len(postings) <= 300:
                for c_idx in postings:
                    cand_scores[c_idx] += 15
                    
    top_cands = sorted(cand_scores.items(), key=lambda x: -x[1])[:15]
    cand_indices = [idx for idx, sc in top_cands]
    cand_eids = [target_eids[idx] for idx in cand_indices]
    
    retrieved_true = set(cand_eids) & true_set
    retrieved_true_count += len(retrieved_true)
    
    if cand_indices:
        feat_matrix = []
        for c_idx in cand_indices:
            cand_data = {
                "name_str": target_names[c_idx],
                "name_toks": target_name_toks[c_idx],
                "name_comp": target_comps[c_idx],
                "addr_str": target_addrs[c_idx],
                "addr_toks": set(target_addrs[c_idx].split()),
                "addr_nums": target_addr_nums[c_idx]
            }
            feat_matrix.append(extract_pairwise_features(s1_data, cand_data))
        probs = booster.predict(np.array(feat_matrix, dtype=np.float32))
        for eid, p in zip(cand_eids, probs):
            if eid in true_set:
                true_match_scores.append(p)
            else:
                false_match_scores.append(p)

print("\n=== RECALL & SCORE RESULTS ===", flush=True)
print(f"Total True Matches: {total_true_count}", flush=True)
print(f"Retrieved in Top-15 Candidates: {retrieved_true_count} ({retrieved_true_count/max(1, total_true_count)*100:.2f}%)", flush=True)

if true_match_scores:
    print(f"True Match Probs: min={np.min(true_match_scores):.4f}, median={np.median(true_match_scores):.4f}, mean={np.mean(true_match_scores):.4f}, p10={np.percentile(true_match_scores, 10):.4f}, p25={np.percentile(true_match_scores, 25):.4f}", flush=True)
if false_match_scores:
    print(f"False Match Probs: max={np.max(false_match_scores):.4f}, median={np.median(false_match_scores):.4f}, mean={np.mean(false_match_scores):.4f}, p90={np.percentile(false_match_scores, 90):.4f}, p99={np.percentile(false_match_scores, 99):.4f}", flush=True)
