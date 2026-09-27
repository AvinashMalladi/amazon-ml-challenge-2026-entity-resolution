import sys
sys.path.append("scripts")
from test_recall_coverage import (
    s1_sample, gt_map, tgt_data, parse_record,
    tok_index, comp_index, num_index, sig_index
)
from rapidfuzz import fuzz
import numpy as np
from collections import defaultdict

print("=== Evaluating Multi-Channel Candidates with Precision Matcher ===")

candidate_matches = []
total_cands = 0

for s1_r in s1_sample.iter_rows(named=True):
    s1_id = s1_r["entity_id"]
    c1 = s1_r["country"]
    n1_str, n1_toks, n1_comp, a1_str, a1_toks, a1_nums, a1_sigs = parse_record(s1_r["business_name"], s1_r["business_address"])
    
    cand_ids = set()
    for t in n1_toks:
        cand_ids.update(tok_index.get((c1, t), [])[:100])
    if len(n1_comp) >= 5:
        cand_ids.update(comp_index.get((c1, n1_comp[:8]), [])[:100])
    for n in a1_nums:
        if len(n) >= 2:
            cand_ids.update(num_index.get((c1, n), [])[:100])
    for sig in a1_sigs:
        cand_ids.update(sig_index.get((c1, sig), [])[:100])
        
    total_cands += len(cand_ids)
    s1_sub_nums = {n for n in a1_nums if len(n) >= 2}
    
    for tid in cand_ids:
        td = tgt_data[tid]
        n2_str = td["n_str"]
        n2_comp = td["n_comp"]
        n2_toks = td["n_toks"]
        a2_str = td["a_str"]
        a2_nums = td["a_nums"]
        has_both_addr = bool(a1_str and a2_str)
        
        t_sub_nums = {n for n in a2_nums if len(n) >= 2}
        common_sub_nums = s1_sub_nums & t_sub_nums
        num_conflict = bool(s1_sub_nums and t_sub_nums and not common_sub_nums)
        
        name_sim = fuzz.token_sort_ratio(n1_str, n2_str) if (n1_str and n2_str) else 0
        addr_sim = fuzz.token_sort_ratio(a1_str, a2_str) if has_both_addr else 0
        
        # 1. Target Address Empty
        if not has_both_addr:
            if name_sim >= 85 and len(n1_str) >= 7 and len(n2_str) >= 7:
                candidate_matches.append((name_sim, s1_id, tid))
            elif n1_toks and n2_toks:
                w1 = next(iter(n1_toks))
                w2 = next(iter(n2_toks))
                if len(w1) >= 5 and len(w2) >= 5 and fuzz.ratio(w1, w2) >= 85:
                    candidate_matches.append((name_sim, s1_id, tid))
            continue
            
        # 2. Both Have Address: Conflict Gate
        if num_conflict and name_sim < 88:
            continue
        if not common_sub_nums and addr_sim < 55:
            continue
            
        compact_match = False
        if len(n1_comp) >= 5 and len(n2_comp) >= 5:
            compact_match = (n1_comp in n2_comp) or (n2_comp in n1_comp)
            
        first_token_match = False
        if n1_str and n2_str:
            t1_first = n1_str.split()[0]
            t2_first = n2_str.split()[0]
            if len(t1_first) >= 4 and t1_first == t2_first:
                first_token_match = True
                
        is_match = False
        score = 0
        
        # Anchor A: High Name Match
        if name_sim >= 80:
            if addr_sim >= 50 or bool(common_sub_nums):
                is_match = True
                score = name_sim + (25 if common_sub_nums else 0) + (addr_sim * 0.2)
        elif compact_match:
            if addr_sim >= 50 or bool(common_sub_nums):
                is_match = True
                score = 85 + (25 if common_sub_nums else 0) + (addr_sim * 0.2)
        # Anchor B: Modified Name + Substantive Address Match
        elif (name_sim >= 55 or first_token_match) and addr_sim >= 60 and bool(common_sub_nums):
            is_match = True
            score = addr_sim + (name_sim * 0.5)
        # Anchor C: Very High Address Match with common substantive numbers
        elif addr_sim >= 85 and bool(common_sub_nums) and (name_sim >= 35 or first_token_match):
            is_match = True
            score = addr_sim + (name_sim * 0.5)
            
        if is_match:
            candidate_matches.append((score, s1_id, tid))

print(f"Total candidates evaluated: {total_cands} (Avg {total_cands/len(s1_sample):.1f}/S1)")
print(f"Total candidate matches passed: {len(candidate_matches)}")

# Global Best-Match Assignment (1-to-many from S1 to targets)
candidate_matches.sort(key=lambda x: -x[0])

target_claimed = {}
s1_final_preds = defaultdict(set)

for score, s1_id, tid in candidate_matches:
    if tid not in target_claimed:
        target_claimed[tid] = s1_id
        s1_final_preds[s1_id].add(tid)

# Evaluate
scores = []
tp_tot, fp_tot, fn_tot = 0, 0, 0

for s1_r in s1_sample.iter_rows(named=True):
    s1_id = s1_r["entity_id"]
    true_set = gt_map.get(s1_id, set())
    pred_set = s1_final_preds[s1_id]
    
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

print(f"\n==========================================")
print(f"GOLDEN PIPELINE (Macro F0.5): {macro_f05:.5f}")
print(f"Precision: {prec*100:.2f}% (TP={tp_tot}, FP={fp_tot})")
print(f"Recall   : {rec*100:.2f}% (FN={fn_tot})")
print(f"==========================================")
