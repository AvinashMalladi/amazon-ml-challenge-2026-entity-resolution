import sys
import polars as pl
import re
import anyascii
from rapidfuzz import fuzz
from collections import defaultdict
import numpy as np

print("=== Evaluating Golden Multi-Anchor Pipeline ===")

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

print(f"Loaded {len(s1_sample)} S1 entities, {len(df_tgt)} targets.")

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
    
    return n_str, set(n_tokens), n_compact, a_str, a_nums

# Parse targets
tgt_data = {}
tok_index = defaultdict(list)
comp_index = defaultdict(list)
num_index = defaultdict(list)

for r in df_tgt.iter_rows(named=True):
    tid = r["entity_id"]
    c = r["country"]
    n_str, n_toks, n_comp, a_str, a_nums = parse_record(r["business_name"], r["business_address"])
    tgt_data[tid] = {
        "eid": tid, "country": c,
        "n_str": n_str, "n_toks": n_toks, "n_comp": n_comp,
        "a_str": a_str, "a_nums": a_nums,
        "has_addr": bool(a_str)
    }
    for t in n_toks:
        tok_index[(c, t)].append(tid)
    if len(n_comp) >= 6:
        comp_index[(c, n_comp[:10])].append(tid)
    for n in a_nums:
        if len(n) >= 3:
            num_index[(c, n)].append(tid)

print("Target indices built. Scoring pairs...")

candidate_matches = []

for s1_r in s1_sample.iter_rows(named=True):
    s1_id = s1_r["entity_id"]
    c1 = s1_r["country"]
    n1_str, n1_toks, n1_comp, a1_str, a1_nums = parse_record(s1_r["business_name"], s1_r["business_address"])
    
    cand_ids = set()
    for t in n1_toks:
        cand_ids.update(tok_index.get((c1, t), [])[:100])
    if len(n1_comp) >= 6:
        cand_ids.update(comp_index.get((c1, n1_comp[:10]), [])[:100])
    for n in a1_nums:
        if len(n) >= 3:
            cand_ids.update(num_index.get((c1, n), [])[:100])
            
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
            if name_sim >= 85 and len(n1_str) >= 8 and len(n2_str) >= 8:
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
        if not common_sub_nums and addr_sim < 50:
            continue
            
        compact_match = False
        if len(n1_comp) >= 6 and len(n2_comp) >= 6:
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
            if addr_sim >= 45 or bool(common_sub_nums):
                is_match = True
                score = name_sim + (20 if common_sub_nums else 0) + (addr_sim * 0.2)
        elif compact_match:
            if addr_sim >= 45 or bool(common_sub_nums):
                is_match = True
                score = 85 + (20 if common_sub_nums else 0) + (addr_sim * 0.2)
        # Anchor B: Modified Name + Substantive Address Match
        elif (name_sim >= 60 or first_token_match) and addr_sim >= 65:
            is_match = True
            score = addr_sim + (name_sim * 0.4)
        # Anchor C: High Address Match (Transliteration / DBA)
        elif addr_sim >= 82 and len(a1_str) >= 20 and len(a2_str) >= 15:
            is_match = True
            score = addr_sim + (name_sim * 0.2)
            
        if is_match:
            candidate_matches.append((score, s1_id, tid))

print(f"Total candidate match pairs: {len(candidate_matches)}")

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
