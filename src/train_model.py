"""
Training Pipeline for Amazon ML Challenge 2026: Business Entity Resolution.
Trains a high-precision LightGBM pairwise matching model on multi-channel candidate pairs.
"""

import os
import sys
import polars as pl
import numpy as np
import lightgbm as lgb
from collections import defaultdict, Counter
from array import array
import time

sys.path.append(os.path.join(os.path.dirname(__file__)))
from normalization import clean_name, clean_address
from features import extract_pairwise_features, FEATURE_NAMES

def train():
    print("=" * 80)
    print("Amazon ML Challenge 2026: High-Precision Model Training Pipeline")
    print("=" * 80)
    
    train_dir = "student_resource/dataset/train"
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    N_S1 = 12000
    print(f"Loading Ground Truth pairs ({N_S1:,} S1 entities)...")
    gt = pl.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), separator="\t", n_rows=N_S1)
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
            
    print(f"Loading Source 1 reference records...")
    s1 = pl.read_csv(os.path.join(train_dir, "train_source1.tsv"), separator="\t")
    s1_sample = s1.filter(pl.col("entity_id").is_in(list(gt_map.keys())))
    
    print(f"Loading Target pool (S2 + S3)...")
    s2 = pl.read_csv(os.path.join(train_dir, "train_source2.tsv"), separator="\t", n_rows=150000)
    s3 = pl.read_csv(os.path.join(train_dir, "train_source3.tsv"), separator="\t", n_rows=150000)
    df_tgt = pl.concat([
        s2.filter(pl.col("entity_id").is_in(list(needed_tgts))),
        s3.filter(pl.col("entity_id").is_in(list(needed_tgts))),
        s2, s3
    ]).unique(subset=["entity_id"])
    print(f"Target pool ready: {len(df_tgt):,} records.")
    
    # Precompute Sole-Occupant Address Map
    s1_addr_counts = Counter()
    for r in s1.iter_rows(named=True):
        a = r["business_address"]
        if a and str(a).lower() not in ('none', 'null', 'nan'):
            _, _, _, _, _, a_comp = clean_address(a)
            if len(a_comp) >= 15:
                s1_addr_counts[a_comp] += 1
                
    tgt_data = {}
    tok_index = defaultdict(list)
    comp_index = defaultdict(list)
    num_index = defaultdict(list)
    sig_index = defaultdict(list)
    addr_tok_index = defaultdict(list)
    addr_compact_index = defaultdict(list)
    
    name_tok_freq = Counter()
    addr_tok_freq = Counter()
    
    for r in df_tgt.iter_rows(named=True):
        tid = r["entity_id"]
        c = r["country"]
        n_str, n_toks, n_comp = clean_name(r["business_name"])
        a_str, a_toks, sub_a_toks, a_nums, a_sigs, a_compact = clean_address(r["business_address"])
        tgt_data[tid] = {
            "eid": tid, "country": c,
            "n_str": n_str, "n_toks": n_toks, "n_comp": n_comp,
            "a_str": a_str, "a_toks": a_toks, "sub_a_toks": sub_a_toks,
            "a_nums": a_nums, "a_compact": a_compact, "has_addr": bool(a_str)
        }
        for t in n_toks:
            name_tok_freq[(c, t)] += 1
            tok_index[(c, t)].append(tid)
        if len(n_comp) >= 5:
            comp_index[(c, n_comp[:8])].append(tid)
        for n in a_nums:
            if len(n) >= 2:
                num_index[(c, n)].append(tid)
        for sig in a_sigs:
            sig_index[(c, sig)].append(tid)
        for at in sub_a_toks:
            addr_tok_freq[(c, at)] += 1
            addr_tok_index[(c, at)].append(tid)
        if len(a_compact) >= 15:
            addr_compact_index[(c, a_compact)].append(tid)
            
    print("Extracting pairwise features for positive and negative pairs...")
    X = []
    y = []
    
    for s1_r in s1_sample.iter_rows(named=True):
        s1_id = s1_r["entity_id"]
        true_set = gt_map.get(s1_id, set())
        c1 = s1_r["country"]
        n1_str, n1_toks, n1_comp = clean_name(s1_r["business_name"])
        a1_str, a1_toks, sub_a_toks, a1_nums, a1_sigs, a1_compact = clean_address(s1_r["business_address"])
        is_sole = bool(a1_compact and len(a1_compact) >= 15 and s1_addr_counts.get(a1_compact, 0) == 1)
        s1_d = (n1_str, n1_toks, n1_comp, a1_str, a1_toks, a1_nums, a1_compact, is_sole)
        
        cand_ids = set()
        sorted_toks = sorted(n1_toks, key=lambda t: name_tok_freq.get((c1, t), 0))
        for t in sorted_toks[:4]:
            cand_ids.update(tok_index.get((c1, t), [])[:100])
        if len(n1_comp) >= 5:
            cand_ids.update(comp_index.get((c1, n1_comp[:8]), [])[:50])
        for n in a1_nums:
            if len(n) >= 2:
                cand_ids.update(num_index.get((c1, n), [])[:50])
        for sig in a1_sigs:
            cand_ids.update(sig_index.get((c1, sig), [])[:50])
        sorted_addr_toks = sorted(sub_a_toks, key=lambda at: addr_tok_freq.get((c1, at), 0))
        for at in sorted_addr_toks[:3]:
            cand_ids.update(addr_tok_index.get((c1, at), [])[:50])
        if is_sole:
            cand_ids.update(addr_compact_index.get((c1, a1_compact), []))
            
        for tid in cand_ids:
            td = tgt_data[tid]
            fv = extract_pairwise_features(s1_d, td)
            label = 1 if tid in true_set else 0
            X.append(fv)
            y.append(label)
            
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"Training dataset: {len(X):,} pairs (Positives: {sum(y):,}, Negatives: {len(y)-sum(y):,})")
    
    train_data = lgb.Dataset(X, label=y, feature_name=FEATURE_NAMES)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.85,
        "min_child_samples": 20,
        "verbose": -1
    }
    
    print("Training LightGBM model...")
    booster = lgb.train(params, train_data, num_boost_round=250)
    save_path = os.path.join(models_dir, "matching_lgbm.txt")
    booster.save_model(save_path)
    print(f"Model saved to {save_path} successfully.")

if __name__ == "__main__":
    train()
