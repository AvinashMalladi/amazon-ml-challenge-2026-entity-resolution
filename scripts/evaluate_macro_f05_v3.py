import sys
sys.path.append("code/business_entity_resolution/src")
import polars as pl
from array import array
from collections import defaultdict, Counter
import numpy as np
import lightgbm as lgb
import re
import anyascii
from normalization import LEGAL_SUFFIXES, COMMON_TOKENS, DOMAIN_REGEX, PUNCT_REGEX, ADDR_STOP_WORDS
from features import extract_pairwise_features

def clean_name_opt(raw_name: str):
    if not raw_name:
        return "", set(), ""
    s_asc = anyascii.anyascii(str(raw_name)).lower()
    words = s_asc.split()
    cleaned_words = [DOMAIN_REGEX.sub('', w) for w in words]
    s_clean = PUNCT_REGEX.sub(' ', " ".join(cleaned_words))
    tokens = [w for w in s_clean.split() if len(w) >= 2 and w not in LEGAL_SUFFIXES and w not in COMMON_TOKENS]
    compact = re.sub(r'[^a-z0-9]', '', s_clean)
    return " ".join(tokens), set(tokens), compact

def clean_address_opt(raw_addr: str):
    if not raw_addr or str(raw_addr).lower() == 'none':
        return "", set(), set(), set()
    s_asc = anyascii.anyascii(str(raw_addr)).lower()
    raw_nums = re.findall(r'\d+', s_asc)
    numbers = set(n.lstrip('0') for n in raw_nums if n.lstrip('0'))
    s_clean = PUNCT_REGEX.sub(' ', s_asc)
    tokens = [w for w in s_clean.split() if len(w) >= 3 and w not in ADDR_STOP_WORDS and w not in COMMON_TOKENS]
    signatures = set()
    for n in numbers:
        for t in tokens:
            if len(t) >= 4 and not t.isdigit():
                signatures.add(f"{n}_{t[:4]}")
    return " ".join(tokens), set(tokens), numbers, signatures

# 1. Load Ground Truth sample
N_S1 = 2000
s1_sample = pl.read_csv("student_resource/dataset/train/train_source1.tsv", separator="\t", n_rows=N_S1)
gt_sample = pl.read_csv("student_resource/dataset/train/train_ground_truth.tsv", separator="\t")
s1_eids = set(s1_sample["entity_id"])
gt_filtered = gt_sample.filter(pl.col("source1_entity_id").is_in(list(s1_eids)))

gt_map = {}
all_true_tgt_ids = set()
for r in gt_filtered.iter_rows(named=True):
    s1_id = r["source1_entity_id"]
    m = r["matched_entity_ids"]
    if m:
        tgts = set(x.strip() for x in m.split(",") if x.strip())
        gt_map[s1_id] = tgts
        all_true_tgt_ids.update(tgts)
    else:
        gt_map[s1_id] = set()

# 2. Load Targets
s2_miss = pl.read_csv("student_resource/dataset/train/train_source2.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(all_true_tgt_ids)))
s3_miss = pl.read_csv("student_resource/dataset/train/train_source3.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(all_true_tgt_ids)))
s2_noise = pl.read_csv("student_resource/dataset/train/train_source2.tsv", separator="\t", n_rows=100000)
s3_noise = pl.read_csv("student_resource/dataset/train/train_source3.tsv", separator="\t", n_rows=100000)
targets = pl.concat([s2_noise, s3_noise, s2_miss, s3_miss]).unique(subset=["entity_id"])

# 3. Index Targets
target_eids = list(targets["entity_id"])
target_names = []
target_addrs = []
target_name_toks = []
target_addr_nums = []
target_comps = []
target_countries = list(targets["country"])

name_index = defaultdict(lambda: array('I'))
sig_index = defaultdict(lambda: array('I'))
num_index = defaultdict(lambda: array('I'))
comp_index = defaultdict(lambda: array('I'))
addr_tok_index = defaultdict(lambda: array('I'))
token_freq = Counter()
addr_tok_freq = Counter()

raw_names = targets["business_name"].to_list()
raw_addrs = targets["business_address"].to_list()

for idx in range(len(target_eids)):
    c = target_countries[idx]
    n_str, n_toks, n_comp = clean_name_opt(raw_names[idx])
    a_str, a_toks, a_nums, a_sigs = clean_address_opt(raw_addrs[idx])
    
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

# 4. Generate candidates & LightGBM predictions
booster = lgb.Booster(model_file="models/matching_lgbm.txt")
cached_results = [] # (s1_id, [ (cand_eid, prob, feats) ])

for s1_r in s1_sample.iter_rows(named=True):
    s1_id = s1_r["entity_id"]
    c = s1_r["country"]
    
    n_str, n_toks, n_comp = clean_name_opt(s1_r["business_name"])
    a_str, a_toks, a_nums, a_sigs = clean_address_opt(s1_r["business_address"])
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
                cand_scores[c_idx] += 15
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
                cand_scores[c_idx] += 10
                    
    if not cand_scores:
        cached_results.append((s1_id, []))
        continue
        
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
    
    cand_tuples = []
    for eid, p, f in zip(cand_eids, probs, feats):
        cand_tuples.append((eid, float(p), f))
    cached_results.append((s1_id, cand_tuples))

print("\n=== EVALUATING ENHANCED MULTI-CHANNEL MACRO F0.5 ===", flush=True)

def score_predictions(preds):
    scores = []
    tps, fps, fns = 0, 0, 0
    for s1_id, true_set in gt_map.items():
        pred_set = preds.get(s1_id, set())
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
    return macro_f05, prec, rec, tps, fps, fns

for thresh in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
    for drop_conflict in [False, True]:
        preds = {}
        for s1_id, cand_tuples in cached_results:
            matched = set()
            for eid, p, fv in cand_tuples:
                if p < thresh:
                    continue
                if drop_conflict and fv[12] == 1.0: # num_conflict == 1
                    continue
                matched.add(eid)
            preds[s1_id] = matched
        mf, p, r, tp, fp, fn = score_predictions(preds)
        print(f"Thresh={thresh:.2f}, DropConflict={drop_conflict} -> Macro F0.5={mf:.5f} | Prec={p*100:.2f}%, Rec={r*100:.2f}% (TP={tp}, FP={fp}, FN={fn})", flush=True)
