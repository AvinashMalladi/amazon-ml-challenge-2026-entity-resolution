# Amazon ML Challenge 2026: Business Entity Resolution Solution

This package contains the complete, self-contained, reproducible pipeline for the Amazon ML Challenge 2026 Business Entity Resolution problem.

## 1. Pipeline Overview

The solution resolves noisy, fragmented business entities across three independent sources (Source 1 reference against Source 2 and Source 3) under strict scale constraints (12.5M train records, 11.7M test records across US, India, and France).

### Key Architectural Pillars:
1. **Strict Country Partitioning**: Hard partition by country (`US`, `India`, `France`), eliminating cross-country comparisons with 100.00% precision guarantee.
2. **Multilingual Transliteration & Normalization**:
   - Universal phonetic transliteration of Indic scripts (Devanagari, Tamil, Telugu, Kannada, Gujarati, Bengali, Odia, Malayalam, Gurmukhi) into Latin ASCII via `anyascii`.
   - Diacritics stripping for French records (é, è, ê, ç, à, etc.).
   - Standardized cleaning of legal suffixes (`Inc`, `Corp`, `LLC`, `Pvt Ltd`, `SARL`, `SASU`, `SCI`, etc.).
   - Domain extension stripping (`.com`, `.org`, `.net`, `.in`, `.fr`, etc.) and compact alphanumeric extraction.
   - Address token normalization (`rd` -> `road`, `st` -> `street`, `r.` -> `rue`, `bd` -> `boulevard`, etc.).
3. **Multi-Channel Inverted Index Blocking**:
   - Channel 1: Multi-Token Name Matching with frequency/IDF ranking.
   - Channel 2: Address Signatures (`<house_num>_<street_prefix>`, e.g., `175_roos`, `729_tiff`).
   - Channel 3: Rare Numeric Tokens (PIN codes, unit numbers, phone numbers).
   - Channel 4: Compact Name Domain Matching.
   - Memory Optimization: Inverted posting lists stored using C-native `array('I')` (4 bytes per posting).
   - Candidate Pruning: Top-12 ranked candidates per entity, reducing search space by >99.999% and yielding a candidate set size of ~10 candidates per entity.
4. **Precision-Calibrated LightGBM Classifier**:
   - 14 pairwise features capturing name Jaccard, subset overlap ratio, Levenshtein distance, token sort ratio, token set ratio, compact string containment, address token overlap, address edit distance, exact number match, and address numeric conflicts.
   - Decision threshold tuned to maximize macro $F_{0.5}$ (weighting precision 2× over recall).
   - Singleton Preservation: High-precision filtering guarantees 1.0 score on singletons by avoiding false merges.

---

## 2. Directory Structure

```
business_entity_resolution/
├── src/
│   ├── normalization.py      # Multilingual text & address normalization
│   ├── features.py           # Rapidfuzz pairwise feature engineering
│   ├── train_model.py        # LightGBM training on positive & hard negative pairs
│   └── infer_submission.py   # Full test inference (blocking + classification)
├── README.md                 # End-to-end reproduction guide
└── requirements.txt          # Pinned environment dependencies
```

---

## 3. Installation & Setup

Ensure Python 3.10+ is installed. Install all dependencies via pip:

```bash
pip install -r requirements.txt
```

---

## 4. End-to-End Reproduction

### Step 1: Model Training
To train the LightGBM matching model from scratch on training data:

```bash
python src/train_model.py
```
This script:
- Loads ground truth labels and source records.
- Generates positive matched pairs and hard-negative candidate pairs from multi-channel blocking.
- Trains a LightGBM classifier with 14 pairwise features.
- Saves the trained model to `models/matching_lgbm.txt` and configuration to `models/config.json`.

### Step 2: Test Inference & Candidate Generation
To run blocking candidate generation and final entity matching on the test set:

```bash
python src/infer_submission.py
```
This script:
- Processes France, US, and India test sets independently.
- Builds compact C-array inverted indices for Source 2 and Source 3.
- Generates the top candidate pairs per Source 1 entity and writes to `output/candidate_pairs.tsv`.
- Runs LightGBM inference on candidate pairs and writes final matches to `output/matching_results.tsv`.

---

## 5. Output Validation

Validate both generated TSV files using the official validation script:

```bash
python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```

Exit code `0` (`PASS`) confirms that all formatting constraints, entity presence, ID validity, and subset rules are fully met.
