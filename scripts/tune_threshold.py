import sys
import polars as pl
import lightgbm as lgb
import numpy as np
from collections import defaultdict

booster = lgb.Booster(model_file="models/matching_lgbm.txt")
print("Model loaded. Finding optimal probability threshold for Macro F0.5...")

# We can re-evaluate on the cached validation candidates
from train_and_validate_lgbm import val_s1, tgt_data, gt_map, compute_features, tok_index, comp_index, num_index, sig_index, addr_tok_index, addr_compact_index, name_tok_freq, addr_tok_freq, parse_record, s1_addr_counts

val_pairs = [] # (s1_id, tid, p, fv)
for r in val_s1:
    s1_id = r["entity_id"]
    c1 = r["country"]
    n1_str, n1_toks, n1_comp, a1_str, a1_toks, sub_a_toks, a1_nums, a1_sigs, a1_compact = parse_record(r["business_name"], r["business_address"])
    is_sole = bool(a1_compact and len(a1_compact) >= 15 and s1_addr_counts.get(a1_compact, 0) == 1)
    s1_d = (n1_str, n1_toks, n1_comp, a1_str, a1_toks, a1_nums, a1_compact, is_sole)
    
    cand_ids = set()
    sorted_toks = sorted(n1_toks, key=lambda t: name_tok_freq.get((c1, t), 0))
    for t in sorted_toks[:4]:
        postings = tok_index.get((c1, t), [])
        cand_ids.update(postings[:150] if len(postings) > 250 else postings)
    if len(n1_comp) >= 5:
        cand_ids.update(comp_index.get((c1, n1_comp[:8]), [])[:100])
    for n in a1_nums:
        if len(n) >= 2:
            cand_ids.update(num_index.get((c1, n), [])[:100])
    for sig in a1_sigs:
        cand_ids.update(sig_index.get((c1, sig), [])[:100])
    sorted_addr_toks = sorted(sub_a_toks, key=lambda at: addr_tok_freq.get((c1, at), 0))
    for at in sorted_addr_toks[:3]:
        postings = addr_tok_index.get((c1, at), [])
        cand_ids.update(postings[:200] if len(postings) > 200 else postings)
    if is_sole:
        cand_ids.update(addr_compact_index.get((c1, a1_compact), []))
        
    cand_list = list(cand_ids)
    if not cand_list:
        continue
        
    feats = [compute_features(s1_d, tgt_data[tid]) for tid in cand_list]
    probs = booster.predict(np.array(feats, dtype=np.float32))
    
    for tid, p, fv in zip(cand_list, probs, feats):
        val_pairs.append((s1_id, tid, float(p), fv))

print(f"Cached {len(val_pairs)} validation pairs. Testing thresholds...")

for th in [0.65, 0.70, 0.75, 0.80, 0.82, 0.85, 0.88, 0.90]:
    cand_matches = []
    for s1_id, tid, p, fv in val_pairs:
        if fv[11] == 1.0 and fv[2] < 0.88: # Conflict gate
            continue
        if fv[13] == 1.0: # Sole occupant exact address
            cand_matches.append((10.0 + p, s1_id, tid))
        elif p >= th:
            cand_matches.append((p, s1_id, tid))
            
    cand_matches.sort(key=lambda x: -x[0])
    claimed = {}
    preds = defaultdict(set)
    for score, s1_id, tid in cand_matches:
        if tid not in claimed:
            claimed[tid] = s1_id
            preds[s1_id].add(tid)
            
    scores = []
    tp_tot, fp_tot, fn_tot = 0, 0, 0
    for r in val_s1:
        s1_id = r["entity_id"]
        true_set = gt_map.get(s1_id, set())
        pred_set = preds[s1_id]
        if len(true_set) == 0:
            s = 1.0 if len(pred_set) == 0 else 0.0
        else:
            if len(pred_set) == 0:
                s = 0.0
            else:
                tp = len(pred_set & true_set)
                fp = len(pred_set - true_set)
                fn = len(true_set - pred_set)
                tp_tot += tp
                fp_tot += fp
                fn_tot += fn
                if tp == 0:
                    s = 0.0
                else:
                    prec = tp / len(pred_set)
                    rec = tp / len(true_set)
                    s = (1.25 * prec * rec) / (0.25 * prec + rec)
        scores.append(s)
    macro_f05 = np.mean(scores)
    prec = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) > 0 else 0
    rec = tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) > 0 else 0
    print(f"Threshold {th:.2f} -> Macro F0.5: {macro_f05:.5f} | Prec: {prec*100:.2f}% | Rec: {rec*100:.2f}% (TP={tp_tot}, FP={fp_tot}, FN={fn_tot})")
