import sys
sys.path.append("code/business_entity_resolution/src")
sys.path.append("scripts")
from test_precision_pipeline import s1_sample, s1_predictions, gt_map, tgt_data, parse_record
from rapidfuzz import fuzz

fps = []
for s1_r in s1_sample.iter_rows(named=True):
    s1_id = s1_r["entity_id"]
    preds = s1_predictions[s1_id]
    true_set = gt_map.get(s1_id, set())
    n1_str, n1_toks, n1_comp, a1_str, a1_nums = parse_record(s1_r["business_name"], s1_r["business_address"])
    for tid in preds:
        if tid not in true_set:
            td = tgt_data[tid]
            name_sim = fuzz.token_sort_ratio(n1_str, td["n_str"])
            addr_sim = fuzz.token_sort_ratio(a1_str, td["a_str"])
            fps.append((s1_r, td, name_sim, addr_sim))
            if len(fps) >= 20:
                break
    if len(fps) >= 20:
        break

print(f"Total False Positives sampled: {len(fps)}")
for i, (s1_r, td, ns, asim) in enumerate(fps):
    print(f"\n--- FP {i+1} (NameSim={ns}, AddrSim={asim}) ---")
    print(f"  S1: '{s1_r['business_name']}' | '{s1_r['business_address']}'")
    print(f"  TR: '{td['eid']}' n_str: '{td['n_str']}' | a_str: '{td['a_str']}'")
