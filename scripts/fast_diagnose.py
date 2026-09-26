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

print("=== Fast Entity Resolution Diagnostic & Optimizer ===", flush=True)

# 1. Load Ground Truth
N_S1 = 5000
gt = pl.read_csv("student_resource/dataset/train/train_ground_truth.tsv", separator="\t", n_rows=N_S1)
gt_dict = {}
all_true_targets = set()
for r in gt.iter_rows(named=True):
    s1_id = r["source1_entity_id"]
    m_str = r["matched_entity_ids"]
    if m_str:
        targets = set(x.strip() for x in m_str.split(",") if x.strip())
        gt_dict[s1_id] = targets
        all_true_targets.update(targets)
    else:
        gt_dict[s1_id] = set()

test_s1_ids = set(gt_dict.keys())
s1 = pl.read_csv("student_resource/dataset/train/train_source1.tsv", separator="\t", n_rows=N_S1)
print(f"Loaded {len(s1)} S1 entities. Singletons: {sum(1 for v in gt_dict.values() if not v)}", flush=True)

# 2. Load Target Records (true targets + 100k noise)
print("Loading target pool...", flush=True)
s2 = pl.read_csv("student_resource/dataset/train/train_source2.tsv", separator="\t", n_rows=100000)
s3 = pl.read_csv("student_resource/dataset/train/train_source3.tsv", separator="\t", n_rows=100000)
df_target = pl.concat([s2, s3]).unique(subset=["entity_id"])
print(f"Loaded {len(df_target)} target pool records.", flush=True)

# Ensure all true targets for our 5k S1 are present in the target pool
existing_target_ids = set(df_target["entity_id"])
missing_targets = all_true_targets - existing_target_ids
print(f"Missing true targets to fetch: {len(missing_targets)}", flush=True)
if missing_targets:
    s2_miss = pl.read_csv("student_resource/dataset/train/train_source2.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(missing_targets)))
    s3_miss = pl.read_csv("student_resource/dataset/train/train_source3.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(missing_targets)))
    df_target = pl.concat([df_target, s2_miss, s3_miss]).unique(subset=["entity_id"])
    print(f"Total target records after adding true matches: {len(df_target)}", flush=True)

# 3. Build Index
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

print("Inverted index ready.", flush=True)

# 4. Extract candidates & features once
booster = lgb.Booster(model_file="models/matching_lgbm.txt")
print("Evaluating candidate blocking & scoring once...", flush=True)

cached_predictions = []  # List of (s1_id, [ (cand_eid, prob, raw_features_dict) ])

total_s1 = len(s1)
start_t = time.time()

for s1_idx, s1_r in enumerate(s1.iter_rows(named=True)):
    s1_id = s1_r["entity_id"]
    c = s1_r["country"]
    
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
        cached_predictions.append((s1_id, []))
        continue
        
    top_cands = sorted(cand_scores.items(), key=lambda x: -x[1])[:12]
    cand_indices = [idx for idx, sc in top_cands]
    cand_eids = [target_eids[idx] for idx in cand_indices]
    
    feat_matrix = []
    cands_info = []
    for c_idx in cand_indices:
        cand_data = {
            "name_str": target_names[c_idx],
            "name_toks": target_name_toks[c_idx],
            "name_comp": target_comps[c_idx],
            "addr_str": target_addrs[c_idx],
            "addr_toks": set(target_addrs[c_idx].split()),
            "addr_nums": target_addr_nums[c_idx]
        }
        fv = extract_pairwise_features(s1_data, cand_data)
        feat_matrix.append(fv)
        cands_info.append(fv)
        
    probs = booster.predict(np.array(feat_matrix, dtype=np.float32))
    
    entry_cands = []
    for i in range(len(cand_eids)):
        entry_cands.append((cand_eids[i], float(probs[i]), cands_info[i]))
    cached_predictions.append((s1_id, entry_cands))
    
    if (s1_idx + 1) % 1000 == 0:
        print(f"Processed {s1_idx+1}/{total_s1} entities in {time.time()-start_t:.1f}s", flush=True)

print("Candidate caching complete. Testing decision rules...", flush=True)

def compute_macro_f05(pred_dict, gt_dict):
    scores = []
    tps, fps, fns = 0, 0, 0
    for s1_id, true_set in gt_dict.items():
        pred_set = pred_dict.get(s1_id, set())
        if len(true_set) == 0:
            score = 1.0 if len(pred_set) == 0 else 0.0
        else:
            if len(pred_set) == 0:
                score = 0.0
            else:
                tp = len(pred_set & true_set)
                fp = len(pred_set - true_set)
                fn = len(true_set - pred_set)
                tps += tp
                fps += fp
                fns += fn
                if tp == 0:
                    score = 0.0
                else:
                    prec = tp / len(pred_set)
                    rec = tp / len(true_set)
                    score = (1.25 * prec * rec) / (0.25 * prec + rec)
        scores.append(score)
    macro_f05 = np.mean(scores)
    prec = tps / (tps + fps) if (tps + fps) > 0 else 0
    rec = tps / (tps + fns) if (tps + fns) > 0 else 0
    return macro_f05, prec, rec, tps, fps, fns

# Test pure thresholding
print("\n--- Pure LightGBM Probability Thresholds ---", flush=True)
for thresh in [0.50, 0.60, 0.70, 0.72, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95]:
    preds = {}
    for s1_id, cand_list in cached_predictions:
        matched = {c_eid for c_eid, prob, _ in cand_list if prob >= thresh}
        preds[s1_id] = matched
    mf, p, r, tp, fp, fn = compute_macro_f05(preds, gt_dict)
    print(f"Thresh={thresh:.2f} -> Macro F0.5 = {mf:.5f} | Precision={p*100:.2f}%, Recall={r*100:.2f}% (TP={tp}, FP={fp}, FN={fn})", flush=True)

# Test with Precision Filter Rules:
# FEATURE_NAMES:
# 0: jaccard_name, 1: overlap_name, 2: fuzz_ratio, 3: fuzz_token_sort, 4: fuzz_token_set,
# 5: comp_ratio, 6: comp_contains, 7: jaccard_addr, 8: overlap_addr, 9: addr_ratio,
# 10: addr_token_sort, 11: num_match, 12: num_conflict, 13: has_addr_both
print("\n--- Testing Precision Filtering Rules ---", flush=True)
for thresh in [0.80, 0.85, 0.88, 0.90, 0.92]:
    for drop_num_conflict in [False, True]:
        for min_name_fuzz in [0.0, 0.60, 0.70]:
            preds = {}
            for s1_id, cand_list in cached_predictions:
                matched = set()
                for c_eid, prob, fv in cand_list:
                    if prob < thresh:
                        continue
                    if drop_num_conflict and fv[12] == 1.0: # num_conflict == 1
                        continue
                    if fv[2] < min_name_fuzz and fv[3] < min_name_fuzz: # fuzz_ratio and token_sort < min
                        continue
                    matched.add(c_eid)
                preds[s1_id] = matched
            mf, p, r, tp, fp, fn = compute_macro_f05(preds, gt_dict)
            if mf > 0.80 or thresh >= 0.88:
                print(f"Thresh={thresh:.2f}, DropNumConflict={drop_num_conflict}, MinFuzz={min_name_fuzz:.2f} -> Macro F0.5={mf:.5f} | Prec={p*100:.2f}%, Rec={r*100:.2f}% (FP={fp}, FN={fn})", flush=True)
