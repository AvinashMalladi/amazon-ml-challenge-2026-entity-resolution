from rapidfuzz import fuzz

FEATURE_NAMES = [
    "jaccard_name",
    "overlap_name",
    "fuzz_ratio",
    "fuzz_token_sort",
    "fuzz_token_set",
    "comp_ratio",
    "comp_contains",
    "jaccard_addr",
    "overlap_addr",
    "addr_ratio",
    "addr_token_sort",
    "num_match",
    "num_conflict",
    "has_addr_both"
]

def extract_pairwise_features(s1_data: dict, cand_data: dict):
    # 1. Name Token Features
    n1 = s1_data["name_toks"]
    n2 = cand_data["name_toks"]
    inter_name = len(n1 & n2)
    union_name = len(n1 | n2)
    
    jaccard_name = inter_name / union_name if union_name > 0 else 0.0
    overlap_name = inter_name / min(len(n1), len(n2)) if (n1 and n2) else 0.0
    
    # 2. Name String Similarities
    str1 = s1_data["name_str"]
    str2 = cand_data["name_str"]
    fuzz_ratio = fuzz.ratio(str1, str2) / 100.0 if (str1 and str2) else 0.0
    fuzz_token_sort = fuzz.token_sort_ratio(str1, str2) / 100.0 if (str1 and str2) else 0.0
    fuzz_token_set = fuzz.token_set_ratio(str1, str2) / 100.0 if (str1 and str2) else 0.0
    
    # 3. Compact / Domain Name Features
    c1 = s1_data["name_comp"]
    c2 = cand_data["name_comp"]
    comp_ratio = fuzz.ratio(c1, c2) / 100.0 if (c1 and c2) else 0.0
    comp_contains = 1.0 if (c1 and c2 and len(c1) >= 5 and len(c2) >= 5 and (c1 in c2 or c2 in c1)) else 0.0
    
    # 4. Address Token Features
    a1 = s1_data["addr_toks"]
    a2 = cand_data["addr_toks"]
    has_addr_both = 1.0 if (a1 and a2) else 0.0
    inter_addr = len(a1 & a2)
    union_addr = len(a1 | a2)
    jaccard_addr = inter_addr / union_addr if union_addr > 0 else 0.0
    overlap_addr = inter_addr / min(len(a1), len(a2)) if (a1 and a2) else 0.0
    
    # 5. Address String Similarities
    astr1 = s1_data["addr_str"]
    astr2 = cand_data["addr_str"]
    addr_ratio = fuzz.ratio(astr1, astr2) / 100.0 if (astr1 and astr2) else 0.0
    addr_token_sort = fuzz.token_sort_ratio(astr1, astr2) / 100.0 if (astr1 and astr2) else 0.0
    
    # 6. Numeric / Address Conflict Features
    nums1 = s1_data["addr_nums"]
    nums2 = cand_data["addr_nums"]
    num_match = 1.0 if (nums1 and nums2 and bool(nums1 & nums2)) else 0.0
    num_conflict = 1.0 if (nums1 and nums2 and not bool(nums1 & nums2)) else 0.0
    
    return [
        jaccard_name,
        overlap_name,
        fuzz_ratio,
        fuzz_token_sort,
        fuzz_token_set,
        comp_ratio,
        comp_contains,
        jaccard_addr,
        overlap_addr,
        addr_ratio,
        addr_token_sort,
        num_match,
        num_conflict,
        has_addr_both
    ]
