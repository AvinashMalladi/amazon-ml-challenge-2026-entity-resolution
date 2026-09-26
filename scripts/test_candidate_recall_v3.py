import sys
sys.path.append("code/business_entity_resolution/src")
import polars as pl
from array import array
from collections import defaultdict, Counter
import numpy as np
import re
import anyascii
from normalization import LEGAL_SUFFIXES, COMMON_TOKENS, DOMAIN_REGEX, PUNCT_REGEX, ADDR_STOP_WORDS

def clean_name_v3(raw_name: str):
    if not raw_name:
        return "", set(), "", set()
    s_asc = anyascii.anyascii(str(raw_name)).lower()
    words = s_asc.split()
    cleaned_words = [DOMAIN_REGEX.sub('', w) for w in words]
    s_clean = PUNCT_REGEX.sub(' ', " ".join(cleaned_words))
    tokens = [w for w in s_clean.split() if len(w) >= 2 and w not in LEGAL_SUFFIXES and w not in COMMON_TOKENS]
    compact = re.sub(r'[^a-z0-9]', '', s_clean)
    
    # Prefix / Sub-tokens for typo tolerance
    prefixes = set()
    for t in tokens:
        if len(t) >= 4:
            prefixes.add(t[:4])
            
    return " ".join(tokens), set(tokens), compact, prefixes

def clean_address_v3(raw_addr: str):
    if not raw_addr or str(raw_addr).lower() == 'none':
        return "", set(), set(), set()
    s_asc = anyascii.anyascii(str(raw_addr)).lower()
    # Find all digits even if joined with letters like 4024B or #53/1
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

# Load sample
s1_sample = pl.read_csv("student_resource/dataset/train/train_source1.tsv", separator="\t", n_rows=2000)
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

# Fetch target records
s2_miss = pl.read_csv("student_resource/dataset/train/train_source2.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(all_true_tgt_ids)))
s3_miss = pl.read_csv("student_resource/dataset/train/train_source3.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(all_true_tgt_ids)))
s2_noise = pl.read_csv("student_resource/dataset/train/train_source2.tsv", separator="\t", n_rows=100000)
s3_noise = pl.read_csv("student_resource/dataset/train/train_source3.tsv", separator="\t", n_rows=100000)
targets = pl.concat([s2_noise, s3_noise, s2_miss, s3_miss]).unique(subset=["entity_id"])

# Build v3 index
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
    n_str, n_toks, n_comp, n_pref = clean_name_v3(raw_names[idx])
    a_str, a_toks, a_nums, a_sigs = clean_address_v3(raw_addrs[idx])
    
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

total_true = sum(len(v) for v in gt_map.values())

print("Testing Recall across Top-K candidates...", flush=True)
for max_k in [10, 12, 14, 16, 20]:
    retrieved_true = 0
    total_cands = 0
    for s1_r in s1_sample.iter_rows(named=True):
        s1_id = s1_r["entity_id"]
        c = s1_r["country"]
        true_set = gt_map.get(s1_id, set())
        
        n_str, n_toks, n_comp, n_pref = clean_name_v3(s1_r["business_name"])
        a_str, a_toks, a_nums, a_sigs = clean_address_v3(s1_r["business_address"])
        
        cand_scores = defaultdict(int)
        
        # 1. Compact name prefix
        if len(n_comp) >= 6:
            postings = comp_index.get((c, n_comp[:10]))
            if postings and len(postings) <= 250:
                for c_idx in postings:
                    cand_scores[c_idx] += 40
                    
        # 2. Rare Name tokens
        sorted_toks = sorted(n_toks, key=lambda t: token_freq[(c, t)])
        for t in sorted_toks[:4]:
            postings = name_index.get((c, t))
            if postings and len(postings) <= 250:
                for c_idx in postings:
                    cand_scores[c_idx] += 15
                    
        # 3. Address signatures
        for sig in a_sigs:
            postings = sig_index.get((c, sig))
            if postings and len(postings) <= 250:
                for c_idx in postings:
                    cand_scores[c_idx] += 30
                    
        # 4. Address numbers (with leading zeros stripped, embedded digits found)
        for n in a_nums:
            if len(n) >= 3:
                postings = num_index.get((c, n))
                if postings and len(postings) <= 250:
                    for c_idx in postings:
                        cand_scores[c_idx] += 15
                        
        # 5. Rare Address tokens (e.g. locality, street word, unique city)
        sorted_addr_toks = sorted(a_toks, key=lambda at: addr_tok_freq[(c, at)])
        for at in sorted_addr_toks[:3]:
            postings = addr_tok_index.get((c, at))
            if postings and len(postings) <= 150: # only rare address tokens!
                for c_idx in postings:
                    cand_scores[c_idx] += 10
                    
        if not cand_scores:
            continue
            
        top_cands = sorted(cand_scores.items(), key=lambda x: -x[1])[:max_k]
        cand_indices = [idx for idx, sc in top_cands]
        cand_eids = [target_eids[idx] for idx in cand_indices]
        
        total_cands += len(cand_eids)
        retrieved_true += len(set(cand_eids) & true_set)
        
    avg_cands = total_cands / len(s1_sample)
    recall = retrieved_true / total_true * 100
    print(f"Top-K={max_k:2d} -> Recall: {recall:.2f}% ({retrieved_true}/{total_true}) | Avg Cands/entity: {avg_cands:.2f}", flush=True)
