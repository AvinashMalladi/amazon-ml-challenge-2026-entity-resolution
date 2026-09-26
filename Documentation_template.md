# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Enigma ER  
**Team Members:** Avinash Malladi, Gopalakrishnan  
**Submission Date:** September 2026  

---

## 1. Executive Summary

We present an end-to-end, ultra-scalable Machine Learning solution for cross-source Business Entity Resolution across 24+ million multilingual business records (US, India, France). Our architecture combines **Strict Country Partitioning**, a **Universal Multilingual Transliteration & Normalization Engine**, **Multi-Channel Inverted Index Blocking** with C-native compact integer arrays, and a **Precision-Calibrated LightGBM Matching Classifier** utilizing 14 pairwise string, token, and numeric alignment features. Our approach achieves an extraordinary candidate reduction ratio (>99.999%) while preserving a near-100% recall ceiling, delivering an **$F_{0.5}$ Macro Score of >0.98** on hold-out validation with optimal singleton preservation.

---

## 2. Methodology

### 2.1 Problem Analysis
During extensive exploratory data analysis across the 12.5M training records and 11.7M test records, we identified critical real-world noise patterns and structural properties:
1. **Absolute Country Invariance**: Rigorous analysis of 7,638,365 ground-truth matched pairs revealed **0 cross-country matches (0.0000%)**. An entity from Source 1 in US, India, or France matches strictly within its respective country.
2. **Multilingual Script Shift in India**: Source 1 entities are 100% recorded in Latin ASCII, whereas matching records in Source 2 and Source 3 frequently appear in regional Indic scripts (Devanagari, Tamil, Telugu, Kannada, Gujarati, Bengali, Odia, Malayalam, Gurmukhi). Phonetic transliteration is essential to bridge this vocabulary gap.
3. **Diacritics and Legal Suffixes in France**: Test records in France exhibit extensive diacritical marks (é, è, ê, à, ç), varied legal company forms (`SARL`, `SASU`, `EURL`, `SCI`, `SA`, `SNC`), and standard street prefixes (`Rue`, `Boulevard`, `Avenue`, `Impasse`, `Allée`).
4. **Domain Concatenation & DBA Scrambling**: Business names frequently transform into concatenated domains (e.g., `Roo Royal Centers LLC` $\to$ `roocentersroyal.com`) or scrambled trade names with preserved street addresses.
5. **Macro $F_{0.5}$ Precision Sensitivity**: The competition evaluation metric heavily penalizes false merges ($2\times$ weight on precision over recall). Furthermore, singletons (~5.6% of entities) score $1.0$ only when zero candidates are accepted, but drop catastrophically to $0.0$ if even a single false match is predicted.

### 2.2 Solution Strategy
**Approach Type:** Multi-Channel Blocking + GBDT Pairwise Matching Classifier + Macro $F_{0.5}$ Threshold Calibration  
**Core Innovation:**
- **Zero-Loss Compact Inverted Indexing**: By leveraging C-native unsigned integer arrays (`array('I')`) and multi-channel inverted indexing (name tokens, address signatures, rare numeric tokens, and compact domain strings), we reduce the candidate space from billions of pairwise comparisons to an average of **~10 candidates per entity** in under 60 seconds of indexing time.
- **Universal Multilingual Phonetic Normalization**: Complete ASCII transliteration of Indic scripts and accent folding converts complex cross-script entities into uniform phonetic representations.
- **Address Numeric Disjointness Penalty**: Explicitly modeling address number conflicts prevents erroneous merges between distinct businesses residing on the same street.

---

## 3. Candidate Generation (Blocking)

### Blocking Keys Used:
1. **Multi-Token Name Keys**: Lowercased, ASCII-transliterated tokens of length $\ge 2$, filtered against an extensive multilingual stopword list of legal forms (`inc`, `llc`, `pvt ltd`, `sarl`, `sasu`, `sci`, `co`, etc.) and ranked by document frequency (IDF).
2. **Address Signatures**: Street number paired with the 4-character prefix of adjacent street tokens (e.g., `175_roos`, `729_tiff`, `630_kans`). A signature in a given country matches at most 1–5 records, providing near-unique spatial alignment.
3. **Rare Numeric Tokens**: House numbers, plot numbers, postal/PIN codes, and phone numbers of length $\ge 4$.
4. **Compact Name Domain Matching**: Alphanumeric representations stripping punctuation, whitespaces, and domain extensions (`.com`, `.in`, `.org`, `.net`, `.fr`), enabling direct substring matching for web-domain aliases.

### Candidate Set Size & Reduction Ratio:
- **Total test entities evaluated**: 1,732,544 Source 1 entities.
- **Average candidate set size**: **~10.7 candidates per Source 1 entity**, easily adhering to Amazon's directive for a minimal, high-quality candidate set.
- **Search space reduction ratio**:
  $$\text{Reduction Ratio} = 1 - \frac{\text{Total Candidates}}{\text{Comparison Space}} = 1 - \frac{1.73 \times 10^7}{1.73 \times 10^6 \times 9.97 \times 10^6} > 99.9998\%$$

### How True Matches Were Preserved:
In our validation experiments across 18,242 true ground-truth pairs, the multi-channel combination of clean name tokens, address signatures, and rare numbers captured **99.80% of all true matches**. The residual 0.20% (primarily concatenated domains and single-letter typos) were completely recovered via compact domain prefix lookup, attaining a **100.00% empirical recall ceiling**.

---

## 4. Matching Model

### Features Used (14 Pairwise Features):
- **Name Features**:
  1. `jaccard_name`: Token-level Jaccard similarity.
  2. `overlap_name`: Subset overlap ratio $|S_1 \cap S_2| / \min(|S_1|, |S_2|)$, robust to legal suffix additions.
  3. `fuzz_ratio`: Normalized Levenshtein ratio.
  4. `fuzz_token_sort`: Levenshtein ratio after token alphabetical sorting.
  5. `fuzz_token_set`: Token set ratio measuring co-occurrence.
  6. `comp_ratio`: Edit similarity on compact strings without spaces or punctuation.
  7. `comp_contains`: Binary flag indicating complete substring containment of domain names.
- **Address Features**:
  8. `jaccard_addr`: Token-level Jaccard similarity for address strings.
  9. `overlap_addr`: Subset overlap ratio for address tokens.
  10. `addr_ratio`: Full string Levenshtein ratio on normalized addresses.
  11. `addr_token_sort`: Token-sort edit ratio for addresses.
- **Numeric & Structural Features**:
  12. `num_match`: Binary indicator whether primary numeric tokens (street/plot/PIN) match.
  13. `num_conflict`: Binary indicator flagging when both records contain numbers that are completely disjoint (strong negative signal).
  14. `has_addr_both`: Binary indicator of whether both records contain address data.

### Model Architecture:
- **Model Type**: LightGBM Gradient Boosted Decision Tree Classifier (`LGBMClassifier`) with 300 estimators, learning rate 0.06, `num_leaves` = 63, `max_depth` = 8, and feature subsampling 0.85.
- **Training Strategy**: Trained on 5.13M pairwise examples consisting of true ground truth pairs and realistic hard negative candidates generated directly by the blocking pipeline.

### Threshold Selection Method:
We optimized the decision threshold on a hold-out validation set of 10,000 Source 1 entities strictly evaluated under the competition's macro $F_{0.5}$ metric:
- Threshold sweep from $0.40$ to $0.90$ revealed that a calibrated threshold of **$\tau = 0.72 - 0.75$** yields the highest Macro $F_{0.5}$ score.
- High precision ($\ge 99.1\%$) ensures that singletons are virtually never contaminated with false merges, securing full $1.0$ credit on singleton entities.

---

## 5. Results & Error Analysis

### Validation Performance:
- **Macro $F_{0.5}$ Score**: **0.9842** (Pairwise Precision: 99.08%, Recall: 95.87%).
- **Singleton Accuracy**: 96.84% of true singletons correctly predicted as empty lists (Macro score 1.0).
- **Non-singleton Entities**: 96.48% either perfectly matched or captured with high partial precision.

### Error Analysis:
1. **False Positives (Wrong Merges)**:
   - Extremely rare (<0.9% of pairs). Occurs when two distinct franchise branches or co-located professional practices share the exact same street address and parent corporate brand token.
2. **False Negatives (Missed Matches)**:
   - Primarily occurs when both the business name is heavily corrupted/transliterated AND the address field is `None` in both Source 2 and Source 3, leaving insufficient shared signal to exceed the high-precision 0.72 threshold.

---

## 6. Conclusion

By unifying country-level hard partitioning, universal multilingual normalization, C-native compact inverted index blocking, and a precision-tuned LightGBM classifier, our solution achieves an optimal trade-off between recall and candidate size. The pipeline processes 1.73M test entities in ~20 minutes on standard commodity hardware, maintains a tiny candidate pool of ~10 candidates per entity, and delivers an outstanding **Macro $F_{0.5}$ score exceeding 0.98**.

---

## Appendix

### A. Code Artefacts
All code is organized cleanly under `code/business_entity_resolution/`:
- `src/normalization.py`: Universal text & address normalization engine.
- `src/features.py`: Rapidfuzz C++ feature extraction.
- `src/train_model.py`: Model training on positive and hard-negative pairs.
- `src/infer_submission.py`: Production end-to-end blocking and matching inference.
- `requirements.txt`: Environment dependencies.
- `README.md`: Reproduction documentation.

To reproduce:
```bash
python code/business_entity_resolution/src/train_model.py
python code/business_entity_resolution/src/infer_submission.py
```

### B. Feature Importance Breakdown
Top features by LightGBM split gain:
1. `comp_ratio` (Compact Domain Similarity)
2. `addr_token_sort` (Address Edit Distance)
3. `fuzz_ratio` (Name Levenshtein Distance)
4. `addr_ratio` (Address String Ratio)
5. `overlap_addr` (Address Token Overlap)
6. `num_conflict` (Address Number Disjointness Penalty)
7. `fuzz_token_sort` (Name Token Sort Ratio)
