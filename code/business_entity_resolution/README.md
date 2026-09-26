# Business Entity Resolution Pipeline

**Team Name:** CodeCanvas  
**Challenge:** Amazon ML Challenge 2026: Business Entity Resolution  

---

## 1. Overview
This package contains the complete, self-contained implementation of the Business Entity Resolution pipeline developed by Team CodeCanvas. The pipeline matches Source 1 business records against Source 2 and Source 3 across US, India, and France.

### Key Highlights:
- **Zero-Loss Multi-Channel Blocking**: Employs C-native integer inverted indices over normalized name tokens, compact alphanumeric representations, address signatures, street numbers, and rare address tokens.
- **Universal Multilingual Transliteration**: Normalizes 9 Indic scripts and French diacritical characters to uniform ASCII representations.
- **Precision-Calibrated LightGBM Matching**: Pairwise classification using 14 string, token, and numeric alignment features with calibrated thresholding and numeric conflict rejection to maximize Macro $F_{0.5}$.

---

## 2. Directory Structure
```
code/business_entity_resolution/
├── src/
│   ├── normalization.py      # Multilingual transliteration and address normalization
│   ├── features.py           # 14 pairwise string and numeric similarity features
│   ├── train_model.py        # LightGBM training script
│   └── infer_submission.py   # Multi-country streaming inference pipeline
├── requirements.txt          # Python dependencies
└── README.md                 # Reproduction documentation
```

---

## 3. Environment Setup
Install the pinned requirements using Python 3.9+:
```bash
pip install -r requirements.txt
```

---

## 4. End-to-End Execution

### Step 1: Model Training (Optional, pre-trained model provided)
To train the LightGBM classifier on the training set pairs:
```bash
python src/train_model.py
```
This saves the trained model to `models/matching_lgbm.txt`.

### Step 2: Full Inference
To run candidate generation (blocking) and matching inference on the test set across all countries (France, US, India):
```bash
python src/infer_submission.py
```
This produces two tab-separated files in the `output/` directory:
1. `output/matching_results.tsv` — Final predicted entity matches (scored on leaderboard).
2. `output/candidate_pairs.tsv` — Candidate blocking set (evaluated for candidate generation quality).

---

## 5. Validation
To verify formatting and integrity using the official validator:
```bash
python ../../student_resource/utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../student_resource/dataset/test
```
Exit code `0` confirms the submission strictly satisfies all competition formatting and integrity constraints.
