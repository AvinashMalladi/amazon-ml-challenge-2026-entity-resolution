import sys
sys.path.append("code/business_entity_resolution/src")
import polars as pl
from normalization import clean_name, clean_address
from rapidfuzz import fuzz

gt = pl.read_csv("student_resource/dataset/train/train_ground_truth.tsv", separator="\t", n_rows=10000)
gt_map = {r["source1_entity_id"]: set(x.strip() for x in r["matched_entity_ids"].split(",")) if r["matched_entity_ids"] else set() for r in gt.iter_rows(named=True)}

s1 = pl.read_csv("student_resource/dataset/train/train_source1.tsv", separator="\t")
s1 = s1.filter(pl.col("entity_id").is_in(list(gt_map.keys())))

s2 = pl.read_csv("student_resource/dataset/train/train_source2.tsv", separator="\t", n_rows=50000)
s3 = pl.read_csv("student_resource/dataset/train/train_source3.tsv", separator="\t", n_rows=50000)
tgt_recs = {}
for df in [s2, s3]:
    for r in df.iter_rows(named=True):
        tid = r["entity_id"]
        n_str, n_toks, _ = clean_name(r["business_name"])
        a_str, a_toks, a_nums, _ = clean_address(r["business_address"])
        tgt_recs[tid] = {"country": r["country"], "name": r["business_name"], "n_str": n_str, "a_str": a_str, "a_nums": a_nums}

fps = []
for r in s1.iter_rows(named=True):
    s1_id = r["entity_id"]
    c1 = r["country"]
    n1_str, _, _ = clean_name(r["business_name"])
    a1_str, _, a1_nums, _ = clean_address(r["business_address"])
    true_set = gt_map[s1_id] & set(tgt_recs.keys())
    for tid, tr in tgt_recs.items():
        if tr["country"] != c1 or tid in true_set:
            continue
        n2_str = tr["n_str"]
        a2_str = tr["a_str"]
        a2_nums = tr["a_nums"]
        name_sim = fuzz.token_sort_ratio(n1_str, n2_str) if (n1_str and n2_str) else 0
        has_both_addr = bool(a1_str and a2_str)
        addr_sim = fuzz.token_sort_ratio(a1_str, a2_str) if has_both_addr else 0
        num_common = len(a1_nums & a2_nums) if (a1_nums and a2_nums) else 0
        num_conflict = len(a1_nums ^ a2_nums) > 0 and len(a1_nums) > 0 and len(a2_nums) > 0 and num_common == 0
        
        matched = False
        rule = ""
        if name_sim >= 75:
            if not has_both_addr or (not num_conflict and addr_sim >= 40):
                matched = True
                rule = "Name>=75"
        elif has_both_addr and num_common > 0 and not num_conflict and addr_sim >= 75:
            if any(len(num) >= 2 for num in (a1_nums & a2_nums)) and len(a1_str) >= 15:
                matched = True
                rule = "Addr>=75"
        if matched:
            fps.append((rule, r["business_name"], tr["name"], r["business_address"], tr["a_str"], name_sim, addr_sim))
            if len(fps) >= 15:
                break
    if len(fps) >= 15:
        break

for rule, n1, n2, a1, a2, ns, asim in fps:
    print(f"[{rule}] S1: '{n1}' | '{a1}'")
    print(f"        TR: '{n2}' | '{a2}'")
    print(f"        ns={ns}, asim={asim}\n")
