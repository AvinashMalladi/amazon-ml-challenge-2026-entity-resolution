import sys
sys.path.append("code/business_entity_resolution/src")
import polars as pl
from collections import defaultdict
import numpy as np
import lightgbm as lgb
from normalization import clean_name, clean_address
from features import extract_pairwise_features

print("=== Evaluating Address Anchored Rule on Ground Truth Sample ===")

N_S1 = 2500
gt = pl.read_csv("student_resource/dataset/train/train_ground_truth.tsv", separator="\t", n_rows=N_S1)
s1 = pl.read_csv("student_resource/dataset/train/train_source1.tsv", separator="\t")
s2 = pl.read_csv("student_resource/dataset/train/train_source2.tsv", separator="\t", n_rows=200000)
s3 = pl.read_csv("student_resource/dataset/train/train_source3.tsv", separator="\t", n_rows=200000)

s1_sample = s1.filter(pl.col("entity_id").is_in(gt["source1_entity_id"].to_list()))
gt_map = {}
needed_tgts = set()
for r in gt.iter_rows(named=True):
    s1_id = r["source1_entity_id"]
    m = r["matched_entity_ids"]
    if m:
        tgts = set(x.strip() for x in m.split(",") if x.strip())
        gt_map[s1_id] = tgts
        needed_tgts.update(tgts)
    else:
        gt_map[s1_id] = set()

# Ensure all true targets exist in pool
missing_tgts = needed_tgts - set(s2["entity_id"]) - set(s3["entity_id"])
if missing_tgts:
    s2_m = pl.read_csv("student_resource/dataset/train/train_source2.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(missing_tgts)))
    s3_m = pl.read_csv("student_resource/dataset/train/train_source3.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(missing_tgts)))
    df_tgt = pl.concat([s2, s3, s2_m, s3_m]).unique(subset=["entity_id"])
else:
    df_tgt = pl.concat([s2, s3]).unique(subset=["entity_id"])

print(f"Target pool size: {len(df_tgt)}")

from array import array
from collections import Counter

target_eids = list(df_tgt["entity_id"])
target_names = []
target_addrs = []
target_name_toks = []
target_addr_nums = []
target_comps = []
target_countries = list(df_tgt["country"])

name_index = defaultdict(lambda: array('I'))
sig_index = defaultdict(lambda: array('I'))
num_index = defaultdict(lambda: array('I'))
comp_index = defaultdict(lambda: array('I'))
addr_tok_index = defaultdict(lambda: array('I'))

token_freq = Counter()
addr_tok_freq = Counter()

raw_names = df_tgt["business_name"].to_list()
raw_addrs = df_tgt["business_address"].to_list()

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
        if len(n) >= 3:
            num_index[(c, n)].append(idx)
    if len(n_comp) >= 6:
        comp_index[(c, n_comp[:10])].append(idx)
    for at in a_toks:
        addr_tok_freq[(c, at)] += 1
        addr_tok_index[(c, at)].append(idx)

booster = lgb.Booster(model_file="models/matching_lgbm.txt")

# Run inference with dual acceptance:
# Accept if prob >= threshold AND num_conflict == 0 AND:
# (has_name_sim OR has_strong_addr)
for thresh in [0.55, 0.60, 0.65, 0.70]:
    scores = []
    tps, fps, fns = 0, 0, 0
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
        if len(n_comp) >= 6:
            postings = comp_index.get((c, n_comp[:10]))
            if postings and len(postings) <= 250:
                for c_idx in postings:
                    cand_scores[c_idx] += 40
        sorted_toks = sorted(n_toks, key=lambda t: token_freq[(c, t)])
        for t in sorted_toks[:4]:
            postings = name_index.get((c, t))
            if postings and len(postings) <= 250:
                for c_idx in postings:
                    cand_scores[c_idx] += 18
        for sig in a_sigs:
            postings = sig_index.get((c, sig))
            if postings and len(postings) <= 250:
                for c_idx in postings:
                    cand_scores[c_idx] += 30
        for n in a_nums:
            if len(n) >= 3:
                postings = num_index.get((c, n))
                if postings and len(postings) <= 250:
                    for c_idx in postings:
                        cand_scores[c_idx] += 15
        sorted_addr_toks = sorted(a_toks, key=lambda at: addr_tok_freq[(c, at)])
        for at in sorted_addr_toks[:3]:
            postings = addr_tok_index.get((c, at))
            if postings and len(postings) <= 150:
                for c_idx in postings:
                    cand_scores[c_idx] += 12
                    
        if not cand_scores:
            pred_set = set()
        else:
            top_cands = sorted(cand_scores.items(), key=lambda x: -x[1])[:16]
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
            
            pred_set = set()
            for eid, p, fv in zip(cand_eids, probs, feats):
                if p < thresh:
                    continue
                if fv[12] == 1.0: # num_conflict == 1
                    continue
                # Dual verification: EITHER name similarity OR strong address anchor (DBA / regional script)
                has_name_sim = (fv[0] > 0.0 or fv[2] >= 0.50 or fv[3] >= 0.50 or fv[5] >= 0.60)
                has_addr_anchor = (fv[11] == 1.0 and (fv[7] >= 0.25 or fv[9] >= 0.60))
                if not (has_name_sim or has_addr_anchor):
                    continue
                pred_set.add(eid)
                
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
    print(f"Thresh={thresh:.2f} -> Macro F0.5={macro_f05:.5f} | Prec={prec*100:.2f}%, Rec={rec*100:.2f}% (TP={tps}, FP={fps}, FN={fns})")
