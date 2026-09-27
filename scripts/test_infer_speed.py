import sys
sys.path.append("code/business_entity_resolution/src")
import polars as pl
import lightgbm as lgb
import time
import os

print("=== Benchmarking Production Inference Speed on Test Set ===")
booster = lgb.Booster(model_file="models/matching_lgbm.txt")
print("Model loaded successfully.")
