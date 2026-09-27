import sys
sys.path.append("code/business_entity_resolution/src")
sys.path.append("scripts")
from test_global_assignment import s1_sample, s1_final_preds, gt_map, tgt_data, parse_record
from rapidfuzz import fuzz

fns = []
for s1_r in s1_sample.iter_rows(named=True):
    s1_id = s1_r["entity_id"]
    true_set = gt_map.get(s1_id, set())
    pred_set = s1_final_preds[s1_id]
    missed = true_set - pred_set
    if missed:
        n1_str, n1_toks, n1_comp, a1_str, a1_nums = parse_record(s1_r["business_name"], s1_r["business_address"])
        for tid in missed:
            if tid in tgt_data:
                td = tgt_data[tid]
                name_sim = fuzz.token_sort_ratio(n1_str, td["n_str"])
                addr_sim = fuzz.token_sort_ratio(a1_str, td["a_str"])
                fns.append((s1_r, td, name_sim, addr_sim))
                if len(fns) >= 25:
                    break
    if len(fns) >= 25:
        break

print(f"Total False Negatives sampled: {len(fns)}")
for i, (s1_r, td, ns, asim) in enumerate(fns):
    print(f"\n--- FN {i+1} (NameSim={ns:.1f}, AddrSim={asim:.1f}) ---")
    print(f"  S1: '{s1_r['business_name']}' | '{s1_r['business_address']}'")
    print(f"  TR: '{td['eid']}' n_str: '{td['n_str']}' | a_str: '{td['a_str']}'")
