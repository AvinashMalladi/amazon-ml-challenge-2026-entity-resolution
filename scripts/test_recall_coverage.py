import sys
import polars as pl
import re
import anyascii
from rapidfuzz import fuzz
from collections import defaultdict
import numpy as np

print("=== Measuring Candidate Recall & Optimal Matching ===")

N_S1 = 2000
gt = pl.read_csv("student_resource/dataset/train/train_ground_truth.tsv", separator="\t", n_rows=N_S1)
s1 = pl.read_csv("student_resource/dataset/train/train_source1.tsv", separator="\t")
s2 = pl.read_csv("student_resource/dataset/train/train_source2.tsv", separator="\t")
s3 = pl.read_csv("student_resource/dataset/train/train_source3.tsv", separator="\t")

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

df_tgt = pl.concat([
    s2.filter(pl.col("entity_id").is_in(list(needed_tgts))),
    s3.filter(pl.col("entity_id").is_in(list(needed_tgts))),
    s2.head(60000), s3.head(60000)
]).unique(subset=["entity_id"])

DOMAIN_REGEX = re.compile(r'\.(com|org|net|in|co|biz|info|io|fr)(\.[a-z]{2})?$', re.IGNORECASE)
LEGAL_SET = {
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'llp', 'ltd', 'limited', 
    'pvt', 'private', 'co', 'company', 'services', 'enterprises', 'solutions', 'holdings',
    'group', 'technologies', 'technology', 'tech', 'ventures', 'international', 'intl',
    'management', 'consulting', 'consultancy', 'consultants', 'associates', 'assoc',
    'center', 'centre', 'industries', 'global', 'systems', 'system', 'partners',
    'sarl', 'sasu', 'sas', 'eurl', 'sci', 'sa', 'snc',
    'praivet', 'limitted', 'elelpi', 'emtrpraijej', 'solyusms', 'holldingg'
}

def parse_record(name, addr):
    n_raw = "" if not name or str(name).lower() in ('none', 'null', 'nan') else str(name)
    a_raw = "" if not addr or str(addr).lower() in ('none', 'null', 'nan') else str(addr)
    
    n_asc = anyascii.anyascii(n_raw).lower()
    n_words = [DOMAIN_REGEX.sub('', w) for w in n_asc.split()]
    n_nodom = " ".join(n_words)
    if " dba " in n_nodom or " d.b.a. " in n_nodom:
        n_nodom = " ".join(re.split(r' d\.?b\.?a\.? ', n_nodom))
    if " formerly: " in n_nodom:
        n_nodom = " ".join(re.split(r' formerly: ', n_nodom))
        
    n_clean = re.sub(r'[^a-z0-9\s]', ' ', n_nodom)
    n_tokens = [w for w in n_clean.split() if len(w) >= 2 and w not in LEGAL_SET]
    n_str = " ".join(n_tokens)
    n_compact = re.sub(r'[^a-z0-9]', '', n_clean)
    
    a_asc = anyascii.anyascii(a_raw).lower()
    a_clean = re.sub(r'[^a-z0-9\s]', ' ', a_asc)
    a_words = a_clean.split()
    a_nums = set()
    for w in a_words:
        if w.isdigit():
            a_nums.add(w.lstrip('0') or '0')
        else:
            nums = re.findall(r'\d+', w)
            for num in nums:
                a_nums.add(num.lstrip('0') or '0')
    a_tokens = [w for w in a_words if len(w) >= 2 and not w.isdigit()]
    a_str = " ".join(a_tokens)
    
    # Address signatures: combine first number with first 2 words
    sigs = set()
    first_num = sorted(list(a_nums))[0] if a_nums else None
    if first_num and a_tokens:
        for t in a_tokens[:2]:
            sigs.add(f"{first_num}_{t}")
            
    return n_str, set(n_tokens), n_compact, a_str, set(a_tokens), a_nums, sigs

tgt_data = {}
tok_index = defaultdict(list)
comp_index = defaultdict(list)
num_index = defaultdict(list)
sig_index = defaultdict(list)

for r in df_tgt.iter_rows(named=True):
    tid = r["entity_id"]
    c = r["country"]
    n_str, n_toks, n_comp, a_str, a_toks, a_nums, a_sigs = parse_record(r["business_name"], r["business_address"])
    tgt_data[tid] = {
        "eid": tid, "country": c,
        "n_str": n_str, "n_toks": n_toks, "n_comp": n_comp,
        "a_str": a_str, "a_toks": a_toks, "a_nums": a_nums,
        "has_addr": bool(a_str)
    }
    for t in n_toks:
        tok_index[(c, t)].append(tid)
    if len(n_comp) >= 5:
        comp_index[(c, n_comp[:8])].append(tid)
    for n in a_nums:
        if len(n) >= 2:
            num_index[(c, n)].append(tid)
    for sig in a_sigs:
        sig_index[(c, sig)].append(tid)

print("Target indices built. Checking Candidate Recall...")

retrieved = 0
total_gt = 0
for s1_r in s1_sample.iter_rows(named=True):
    s1_id = s1_r["entity_id"]
    true_set = gt_map.get(s1_id, set())
    if not true_set:
        continue
    total_gt += len(true_set)
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
        
    retrieved += len(true_set & cand_ids)

print(f"Total Ground Truth Matches: {total_gt}")
print(f"Candidates Retrieved: {retrieved} ({retrieved/total_gt*100:.2f}%)")
