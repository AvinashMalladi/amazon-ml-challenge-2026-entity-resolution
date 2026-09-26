"""
Training Pipeline for Amazon ML Challenge 2026: Business Entity Resolution.
Trains a high-precision LightGBM pairwise matching model on training dataset pairs.
"""

import os
import sys
import polars as pl
import numpy as np
import lightgbm as lgb
from collections import defaultdict, Counter
from array import array
import time

from normalization import clean_name, clean_address
from features import extract_pairwise_features, FEATURE_NAMES

def train():
    print("=" * 70)
    print("Amazon ML Challenge 2026: Model Training Pipeline")
    print("=" * 70)
    
    train_dir = "student_resource/dataset/train"
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load Ground Truth
    print("Loading Ground Truth pairs...")
    gt = pl.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), separator="\t", n_rows=250000)
    gt_map = {}
    needed_target_ids = set()
    for r in gt.iter_rows(named=True):
        s1_id = r["source1_entity_id"]
        m = r["matched_entity_ids"]
        if m:
            tgts = set(x.strip() for x in m.split(",") if x.strip())
            gt_map[s1_id] = tgts
            needed_target_ids.update(tgts)
        else:
            gt_map[s1_id] = set()
            
    s1_ids = set(gt_map.keys())
    print(f"Loaded {len(gt_map)} S1 records. True targets needed: {len(needed_target_ids)}")
    
    # 2. Load S1 records
    print("Loading Source 1 records...")
    s1 = pl.read_csv(os.path.join(train_dir, "train_source1.tsv"), separator="\t").filter(pl.col("entity_id").is_in(list(s1_ids)))
    
    # 3. Load Targets (S2 + S3)
    print("Loading Source 2 and Source 3 targets...")
    s2 = pl.read_csv(os.path.join(train_dir, "train_source2.tsv"), separator="\t", n_rows=500000)
    s3 = pl.read_csv(os.path.join(train_dir, "train_source3.tsv"), separator="\t", n_rows=500000)
    df_tgt = pl.concat([s2, s3]).unique(subset=["entity_id"])
    
    # Ensure needed true targets are present
    missing_targets = needed_target_ids - set(df_tgt["entity_id"])
    if missing_targets:
        print(f"Loading {len(missing_targets)} missing targets...")
        s2_m = pl.read_csv(os.path.join(train_dir, "train_source2.tsv"), separator="\t").filter(pl.col("entity_id").is_in(list(missing_targets)))
        s3_m = pl.read_csv(os.path.join(train_dir, "train_source3.tsv"), separator="\t").filter(pl.col("entity_id").is_in(list(missing_targets)))
        df_tgt = pl.concat([df_tgt, s2_m, s3_m]).unique(subset=["entity_id"])
        
    print(f"Target pool ready: {len(df_tgt)} records.")
    
    # 4. Feature Extraction & Dataset Building
    target_data = {}
    for r in df_tgt.iter_rows(named=True):
        eid = r["entity_id"]
        n_str, n_toks, n_comp = clean_name(r["business_name"])
        a_str, a_toks, a_nums, a_sigs = clean_address(r["business_address"])
        target_data[eid] = {
            "name_str": n_str, "name_toks": frozenset(n_toks), "name_comp": n_comp,
            "addr_str": a_str, "addr_toks": frozenset(a_toks), "addr_nums": a_nums
        }
        
    X = []
    y = []
    
    print("Extracting features for positive and hard-negative pairs...")
    for s1_r in s1.iter_rows(named=True):
        s1_id = s1_r["entity_id"]
        true_tgts = gt_map.get(s1_id, set())
        
        n_str, n_toks, n_comp = clean_name(s1_r["business_name"])
        a_str, a_toks, a_nums, a_sigs = clean_address(s1_r["business_address"])
        s1_d = {
            "name_str": n_str, "name_toks": frozenset(n_toks), "name_comp": n_comp,
            "addr_str": a_str, "addr_toks": frozenset(a_toks), "addr_nums": a_nums
        }
        
        for tid in true_tgts:
            if tid in target_data:
                X.append(extract_pairwise_features(s1_d, target_data[tid]))
                y.append(1)
                
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"Training dataset: {len(X)} pairs. Positive pairs: {sum(y)}")
    
    train_data = lgb.Dataset(X, label=y, feature_name=FEATURE_NAMES)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "verbose": 1
    }
    
    print("Training LightGBM model...")
    booster = lgb.train(params, train_data, num_boost_round=300)
    
    save_path = os.path.join(models_dir, "matching_lgbm.txt")
    booster.save_model(save_path)
    print(f"Model saved to {save_path} successfully.")

if __name__ == "__main__":
    train()
