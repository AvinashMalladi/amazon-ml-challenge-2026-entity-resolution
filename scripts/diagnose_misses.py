import sys
sys.path.append("code/business_entity_resolution/src")
import polars as pl
from array import array
from collections import defaultdict, Counter
import numpy as np
import lightgbm as lgb
from normalization import clean_name, clean_address

# Load 500 S1 records
s1_sample = pl.read_csv("student_resource/dataset/train/train_source1.tsv", separator="\t", n_rows=500)
gt_sample = pl.read_csv("student_resource/dataset/train/train_ground_truth.tsv", separator="\t")

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

# Fetch target records
s2_miss = pl.read_csv("student_resource/dataset/train/train_source2.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(all_true_tgt_ids)))
s3_miss = pl.read_csv("student_resource/dataset/train/train_source3.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(all_true_tgt_ids)))
s2_noise = pl.read_csv("student_resource/dataset/train/train_source2.tsv", separator="\t", n_rows=50000)
s3_noise = pl.read_csv("student_resource/dataset/train/train_source3.tsv", separator="\t", n_rows=50000)
targets = pl.concat([s2_noise, s3_noise, s2_miss, s3_miss]).unique(subset=["entity_id"])

# Build index
target_eids = list(targets["entity_id"])
eid_to_idx = {eid: idx for idx, eid in enumerate(target_eids)}
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

# Find missed true matches
missed_reasons = Counter()
samples_shown = 0

for s1_r in s1_sample.iter_rows(named=True):
    s1_id = s1_r["entity_id"]
    c = s1_r["country"]
    true_set = gt_map.get(s1_id, set())
    if not true_set:
        continue
        
    n_str, n_toks, n_comp = clean_name(s1_r["business_name"])
    a_str, a_toks, a_nums, a_sigs = clean_address(s1_r["business_address"])
    
    cand_scores = defaultdict(int)
    sorted_toks = sorted(n_toks, key=lambda t: token_freq[(c, t)])
    for t in sorted_toks[:4]:
        postings = name_index.get((c, t))
        if postings and len(postings) <= 250:
            for c_idx in postings:
                cand_scores[c_idx] += 12
    for sig in a_sigs:
        postings = sig_index.get((c, sig))
        if postings and len(postings) <= 250:
            for c_idx in postings:
                cand_scores[c_idx] += 30
    for n in a_nums:
        if len(n) >= 4:
            postings = num_index.get((c, n))
            if postings and len(postings) <= 250:
                for c_idx in postings:
                    cand_scores[c_idx] += 15
                    
    top_cands = sorted(cand_scores.items(), key=lambda x: -x[1])[:12]
    cand_indices = set(idx for idx, sc in top_cands)
    cand_eids = set(target_eids[idx] for idx in cand_indices)
    
    for true_eid in true_set:
        if true_eid not in cand_eids:
            # Why missed?
            if true_eid not in eid_to_idx:
                missed_reasons["not_in_pool"] += 1
                continue
            t_idx = eid_to_idx[true_eid]
            if t_idx not in cand_scores:
                missed_reasons["zero_candidate_score"] += 1
                if samples_shown < 5:
                    print(f"\n[ZERO SCORE MISSED MATCH]")
                    print(f"S1: {s1_r['business_name']} | Addr: {s1_r['business_address']}")
                    print(f"Toks: {n_toks}, Sigs: {a_sigs}, Nums: {a_nums}")
                    print(f"Target ({true_eid}): {raw_names[t_idx]} | Addr: {raw_addrs[t_idx]}")
                    print(f"Target Toks: {target_name_toks[t_idx]}, Target Nums: {target_addr_nums[t_idx]}")
                    samples_shown += 1
            else:
                score = cand_scores[t_idx]
                min_top = top_cands[-1][1] if top_cands else 0
                missed_reasons["ranked_outside_top12"] += 1
                if samples_shown < 10:
                    print(f"\n[OUTSIDE TOP-12 MISSED MATCH (Score={score} vs Top12 min={min_top})]")
                    print(f"S1: {s1_r['business_name']} | Addr: {s1_r['business_address']}")
                    print(f"Target ({true_eid}): {raw_names[t_idx]} | Addr: {raw_addrs[t_idx]}")
                    samples_shown += 1

print("\n=== MISSED REASONS BREAKDOWN ===", flush=True)
print(missed_reasons, flush=True)
