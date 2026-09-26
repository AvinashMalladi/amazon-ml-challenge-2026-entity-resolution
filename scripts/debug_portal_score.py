import sys
sys.path.append("src")
import polars as pl
from array import array
from collections import defaultdict, Counter
import numpy as np
import lightgbm as lgb
import time
from normalization import clean_name, clean_address
from features import extract_pairwise_features

print("=== Simulating Full Portal Evaluation on Train Ground Truth ===")

# Load 10,000 S1 records from train
N_TEST = 10000
gt = pl.read_csv("student_resource/dataset/train/train_ground_truth.tsv", separator="\t", n_rows=N_TEST)
gt_dict = {}
for r in gt.iter_rows(named=True):
    s1_id = r["source1_entity_id"]
    m_str = r["matched_entity_ids"]
    if m_str:
        gt_dict[s1_id] = set(x.strip() for x in m_str.split(",") if x.strip())
    else:
        gt_dict[s1_id] = set()

test_s1_ids = set(gt_dict.keys())
s1 = pl.read_csv("student_resource/dataset/train/train_source1.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(test_s1_ids)))

# Load target records (noise + targets)
all_targets = set()
for s in gt_dict.values():
    all_targets.update(s)

s2_noise = pl.read_csv("student_resource/dataset/train/train_source2.tsv", separator="\t", n_rows=150000)
s3_noise = pl.read_csv("student_resource/dataset/train/train_source3.tsv", separator="\t", n_rows=150000)
s2_tgt = pl.read_csv("student_resource/dataset/train/train_source2.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(all_targets)))
s3_tgt = pl.read_csv("student_resource/dataset/train/train_source3.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(all_targets)))

df_target = pl.concat([s2_noise, s3_noise, s2_tgt, s3_tgt]).unique(subset=["entity_id"])
print(f"Loaded S1={len(s1)}, Targets={len(df_target)}")

booster = lgb.Booster(model_file="models/matching_lgbm.txt")

# Build index exactly like infer_submission.py
target_eids = list(df_target["entity_id"])
target_names = []
target_addrs = []
target_name_toks = []
target_addr_nums = []
target_comps = []
target_countries = list(df_target["country"])

name_index = defaultdict(lambda: array('I'))
sig_index = defaultdict(lambda: array('I'))
num_index = defaultdict(lambda: array('I'))
token_freq = Counter()

raw_names = df_target["business_name"].to_list()
raw_addrs = df_target["business_address"].to_list()

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

print("Index built.")

# Run scoring with different thresholds and strategies
def evaluate(threshold, max_cands=12):
    f05_scores = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for s1_r in s1.iter_rows(named=True):
        s1_id = s1_r["entity_id"]
        c = s1_r["country"]
        true_set = gt_dict[s1_id]
        
        n_str, n_toks, n_comp = clean_name(s1_r["business_name"])
        a_str, a_toks, a_nums, a_sigs = clean_address(s1_r["business_address"])
        s1_data = {
            "name_str": n_str, "name_toks": n_toks, "name_comp": n_comp,
            "addr_str": a_str, "addr_toks": a_toks, "addr_nums": a_nums
        }
        
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
                        
        if not cand_scores:
            pred_set = set()
        else:
            top_cands = sorted(cand_scores.items(), key=lambda x: -x[1])[:max_cands]
            cand_indices = [idx for idx, sc in top_cands]
            cand_eids = [target_eids[idx] for idx in cand_indices]
            
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
            pred_set = {cand_eids[i] for i, p in enumerate(probs) if p >= threshold}
            
        # Official Macro F0.5
        if len(true_set) == 0:
            score = 1.0 if len(pred_set) == 0 else 0.0
        else:
            if len(pred_set) == 0:
                score = 0.0
            else:
                tp = len(pred_set & true_set)
                fp = len(pred_set - true_set)
                fn = len(true_set - pred_set)
                total_tp += tp
                total_fp += fp
                total_fn += fn
                if tp == 0:
                    score = 0.0
                else:
                    prec = tp / len(pred_set)
                    rec = tp / len(true_set)
                    score = (1.25 * prec * rec) / (0.25 * prec + rec)
        f05_scores.append(score)
        
    macro_f05 = np.mean(f05_scores)
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    print(f"Thresh: {threshold:.2f} | Max Cands: {max_cands:2d} -> Macro F_0.5: {macro_f05:.5f} | Prec: {prec*100:.2f}%, Rec: {rec*100:.2f}% (TP={total_tp}, FP={total_fp}, FN={total_fn})")
    return macro_f05

print("\n--- Testing Thresholds ---")
for t in [0.50, 0.60, 0.70, 0.72, 0.75, 0.80, 0.85, 0.90, 0.95]:
    evaluate(t)
