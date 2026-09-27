import sys
sys.path.append("code/business_entity_resolution/src")
import polars as pl
from collections import Counter
import anyascii
import re

print("=== Analyzing All 10,000 Ground Truth Pairs ===")

gt = pl.read_csv("student_resource/dataset/train/train_ground_truth.tsv", separator="\t", n_rows=2500)
s1 = pl.read_csv("student_resource/dataset/train/train_source1.tsv", separator="\t")
s2 = pl.read_csv("student_resource/dataset/train/train_source2.tsv", separator="\t")
s3 = pl.read_csv("student_resource/dataset/train/train_source3.tsv", separator="\t")

s1_map = {r["entity_id"]: r for r in s1.filter(pl.col("entity_id").is_in(gt["source1_entity_id"].to_list())).iter_rows(named=True)}

all_tgts = []
for r in gt.iter_rows(named=True):
    m = r["matched_entity_ids"]
    if m:
        all_tgts.extend([x.strip() for x in m.split(",") if x.strip()])

s2_map = {r["entity_id"]: r for r in s2.filter(pl.col("entity_id").is_in(all_tgts)).iter_rows(named=True)}
s3_map = {r["entity_id"]: r for r in s3.filter(pl.col("entity_id").is_in(all_tgts)).iter_rows(named=True)}

from normalization import clean_name, clean_address
from rapidfuzz import fuzz

total_pairs = 0
name_match_only = 0
addr_match_only = 0
both_match = 0
neither_match = 0

examples_dba = []

for r in gt.iter_rows(named=True):
    s1_id = r["source1_entity_id"]
    m = r["matched_entity_ids"]
    if not m:
        continue
    s1_r = s1_map.get(s1_id)
    if not s1_r:
        continue
        
    n1_str, n1_toks, n1_comp = clean_name(s1_r["business_name"])
    a1_str, a1_toks, a1_nums, a1_sigs = clean_address(s1_r["business_address"])
    
    for tid in [x.strip() for x in m.split(",") if x.strip()]:
        tgt_r = s2_map.get(tid) or s3_map.get(tid)
        if not tgt_r:
            continue
        total_pairs += 1
        n2_str, n2_toks, n2_comp = clean_name(tgt_r["business_name"])
        a2_str, a2_toks, a2_nums, a2_sigs = clean_address(tgt_r["business_address"])
        
        name_sim = fuzz.token_sort_ratio(n1_str, n2_str)
        has_name = bool(n1_toks & n2_toks) or name_sim >= 60 or (n1_comp and n2_comp and n1_comp in n2_comp or n2_comp in n1_comp)
        
        has_addr = bool(a1_sigs & a2_sigs) or (bool(a1_nums & a2_nums) and bool(a1_toks & a2_toks)) or (fuzz.token_sort_ratio(a1_str, a2_str) >= 60)
        
        if has_name and has_addr:
            both_match += 1
        elif has_name and not has_addr:
            name_match_only += 1
        elif not has_name and has_addr:
            addr_match_only += 1
            if len(examples_dba) < 10:
                examples_dba.append((s1_r, tgt_r, n1_str, n2_str, a1_str, a2_str, a1_sigs & a2_sigs, a1_nums & a2_nums))
        else:
            neither_match += 1

print(f"Total True Pairs Evaluated: {total_pairs}")
print(f"Both Name & Address Match : {both_match} ({both_match/total_pairs*100:.2f}%)")
print(f"Name Match Only (No Addr) : {name_match_only} ({name_match_only/total_pairs*100:.2f}%)")
print(f"Address Match Only (DBA!) : {addr_match_only} ({addr_match_only/total_pairs*100:.2f}%)")
print(f"Neither Match             : {neither_match} ({neither_match/total_pairs*100:.2f}%)")

print("\n--- Examples of Address Match Only (DBA Names) ---")
for ex in examples_dba[:5]:
    s1_r, tgt_r, n1, n2, a1, a2, sigs, nums = ex
    print(f"S1 Name: '{s1_r['business_name']}' | Addr: '{s1_r['business_address']}'")
    print(f"Target : '{tgt_r['business_name']}' | Addr: '{tgt_r['business_address']}'")
    print(f"Matching Sigs: {sigs}, Matching Nums: {nums}")
    print()
