"""
High-Precision Feature Extraction Module for Amazon ML Challenge 2026.
Computes 14 discriminative pairwise similarity features across name, address, and numbers.
"""

from rapidfuzz import fuzz

FEATURE_NAMES = [
    "jacc_name",
    "f_ratio",
    "f_tok_sort",
    "f_tok_set",
    "comp_ratio",
    "comp_match",
    "first_match",
    "jacc_addr",
    "a_ratio",
    "a_tok_sort",
    "has_num_match",
    "has_num_conflict",
    "has_both_addr",
    "exact_addr_sole"
]

def extract_pairwise_features(s1_d: tuple, td: dict):
    n1_str, n1_toks, n1_comp, a1_str, a1_toks, a1_nums, a1_compact, is_sole = s1_d
    n2_str = td["n_str"]
    n2_toks = td["n_toks"]
    n2_comp = td["n_comp"]
    a2_str = td["a_str"]
    a2_toks = td["a_toks"]
    a2_nums = td["a_nums"]
    a2_compact = td["a_compact"]
    has_both_addr = bool(a1_str and a2_str)
    
    # 1. Name features
    jacc_name = len(n1_toks & n2_toks) / len(n1_toks | n2_toks) if (n1_toks or n2_toks) else 0.0
    f_ratio = fuzz.ratio(n1_str, n2_str) / 100.0 if (n1_str and n2_str) else 0.0
    f_tok_sort = fuzz.token_sort_ratio(n1_str, n2_str) / 100.0 if (n1_str and n2_str) else 0.0
    f_tok_set = fuzz.token_set_ratio(n1_str, n2_str) / 100.0 if (n1_str and n2_str) else 0.0
    comp_ratio = fuzz.ratio(n1_comp, n2_comp) / 100.0 if (n1_comp and n2_comp) else 0.0
    comp_match = 1.0 if (len(n1_comp) >= 5 and len(n2_comp) >= 5 and (n1_comp in n2_comp or n2_comp in n1_comp)) else 0.0
    
    first_match = 0.0
    if n1_str and n2_str:
        w1 = n1_str.split()[0]
        w2 = n2_str.split()[0]
        if len(w1) >= 4 and w1 == w2:
            first_match = 1.0
            
    # 2. Address features
    jacc_addr = len(a1_toks & a2_toks) / len(a1_toks | a2_toks) if (has_both_addr and (a1_toks or a2_toks)) else 0.0
    a_ratio = fuzz.ratio(a1_str, a2_str) / 100.0 if has_both_addr else 0.0
    a_tok_sort = fuzz.token_sort_ratio(a1_str, a2_str) / 100.0 if has_both_addr else 0.0
    
    s1_sub_nums = {n for n in a1_nums if len(n) >= 2}
    t_sub_nums = {n for n in a2_nums if len(n) >= 2}
    common_sub_nums = s1_sub_nums & t_sub_nums
    has_num_match = 1.0 if bool(common_sub_nums) else 0.0
    has_num_conflict = 1.0 if (s1_sub_nums and t_sub_nums and not common_sub_nums) else 0.0
    exact_addr_sole = 1.0 if (is_sole and a1_compact == a2_compact) else 0.0
    
    return [
        jacc_name, f_ratio, f_tok_sort, f_tok_set, comp_ratio, comp_match, first_match,
        jacc_addr, a_ratio, a_tok_sort, has_num_match, has_num_conflict,
        1.0 if has_both_addr else 0.0, exact_addr_sole
    ]
