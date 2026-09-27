import sys
import polars as pl

print("=== Forensic Inspection of Ground Truth Matching Patterns ===")

gt = pl.read_csv("student_resource/dataset/train/train_ground_truth.tsv", separator="\t", n_rows=100)
s1 = pl.read_csv("student_resource/dataset/train/train_source1.tsv", separator="\t")
s2 = pl.read_csv("student_resource/dataset/train/train_source2.tsv", separator="\t")
s3 = pl.read_csv("student_resource/dataset/train/train_source3.tsv", separator="\t")

s1_dict = {r["entity_id"]: r for r in s1.filter(pl.col("entity_id").is_in(gt["source1_entity_id"].to_list())).iter_rows(named=True)}

all_matched_ids = []
for r in gt.iter_rows(named=True):
    m = r["matched_entity_ids"]
    if m:
        all_matched_ids.extend([x.strip() for x in m.split(",") if x.strip()])

s2_dict = {r["entity_id"]: r for r in s2.filter(pl.col("entity_id").is_in(all_matched_ids)).iter_rows(named=True)}
s3_dict = {r["entity_id"]: r for r in s3.filter(pl.col("entity_id").is_in(all_matched_ids)).iter_rows(named=True)}

print(f"Loaded {len(s1_dict)} S1, {len(s2_dict)} S2, {len(s3_dict)} S3 matched records.\n")

for i, r in enumerate(gt.iter_rows(named=True)):
    s1_id = r["source1_entity_id"]
    m = r["matched_entity_ids"]
    if not m:
        continue
    matched_ids = [x.strip() for x in m.split(",") if x.strip()]
    s1_rec = s1_dict.get(s1_id)
    if not s1_rec:
        continue
        
    print(f"--- [Entity #{i+1}] S1 ID: {s1_id} | Country: {s1_rec['country']} ---")
    print(f"  S1 Name   : '{s1_rec['business_name']}'")
    print(f"  S1 Address: '{s1_rec['business_address']}'")
    print(f"  Total True Matches ({len(matched_ids)}):")
    for mid in matched_ids:
        tgt_rec = s2_dict.get(mid) or s3_dict.get(mid)
        if tgt_rec:
            prefix = mid[:2]
            print(f"    [{prefix}] ID: {mid}")
            print(f"         Name   : '{tgt_rec['business_name']}'")
            print(f"         Address: '{tgt_rec['business_address']}'")
    print()
    if i >= 10:
        break
