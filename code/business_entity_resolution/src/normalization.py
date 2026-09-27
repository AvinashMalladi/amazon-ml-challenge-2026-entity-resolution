"""
High-Precision Entity Normalization for Amazon ML Challenge 2026.
Optimized for multi-language, multi-script business records (US, India, France).
"""

import re
import anyascii

LEGAL_SUFFIXES = {
    # English / Global
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'llp', 'ltd', 'limited', 
    'pvt', 'private', 'co', 'company', 'services', 'enterprises', 'solutions', 'holdings',
    'group', 'technologies', 'technology', 'tech', 'ventures', 'international', 'intl',
    'management', 'consulting', 'consultancy', 'consultants', 'associates', 'assoc',
    'center', 'centre', 'industries', 'global', 'systems', 'system', 'partners',
    # French
    'sarl', 'sasu', 'sas', 'eurl', 'sci', 'sa', 'snc', 'gie', 'selarl', 'scop', 'sem',
    'association', 'asso', 'societe', 'ste', 'etablissement', 'ets', 'cie', 'freres', 'fils',
    # Indic transliterated legal suffixes
    'praivet', 'limitted', 'elelpi', 'emtrpraijej', 'solyusms', 'holldingg', 'pvt.', 'ltd.'
}

ADDR_STOP_WORDS = {
    'road', 'street', 'avenue', 'drive', 'lane', 'place', 'boulevard', 'court', 'way',
    'circle', 'highway', 'parkway', 'suite', 'floor', 'block', 'plot', 'door', 'flat',
    'house', 'building', 'tower', 'complex', 'near', 'opp', 'opposite', 'behind', 'beside',
    'rd', 'st', 'ave', 'dr', 'ln', 'pl', 'blvd', 'ct', 'cir', 'hwy', 'pkwy', 'ste', 'fl',
    'north', 'south', 'east', 'west', 'po', 'box', 'pmb',
    'rue', 'av', 'bd', 'all', 'imp', 'chm', 'rte', 'pl', 'res', 'bat', 'bis', 'ter'
}

COMMON_TOKENS = {
    'the', 'and', 'of', 'in', 'for', 'at', 'on', 'to', 'a', 'an', 'de', 'du', 'des', 'la', 'le', 'les', 'et'
}

GENERIC_FIRST_TOKENS = {
    'sri', 'shri', 'new', 'om', 'dr', 'hotel', 'krishna', 'balaji', 'jai', 'star',
    'royal', 'apex', 'best', 'the', 'saint', 'st', 'mr', 'mrs', 'prof'
}

DOMAIN_REGEX = re.compile(r'\.(com|org|net|in|co|fr|gov|edu|biz|info|io)(\.[a-z]{2})?$', re.IGNORECASE)
PUNCT_REGEX = re.compile(r'[\.\-\_\,\/\:\;\(\)\[\]\<\>\&\"\'\`\+\#\@\*\=\\\~]')

def clean_name(raw_name: str):
    if not raw_name or str(raw_name).strip().lower() in ('none', 'null', 'nan'):
        return "", set(), ""
    
    s_asc = anyascii.anyascii(str(raw_name)).lower()
    
    # Strip domain extensions e.g. "zanderblue.com" -> "zanderblue"
    words = s_asc.split()
    cleaned_words = [DOMAIN_REGEX.sub('', w) for w in words]
    s_nodom = " ".join(cleaned_words)
    
    # Strip DBA / Trade Name prefixes
    if " dba " in s_nodom or " d.b.a. " in s_nodom or " d/b/a " in s_nodom:
        s_nodom = " ".join(re.split(r' d\.?b\.?a\.? | d/b/a ', s_nodom))
    if " formerly: " in s_nodom:
        s_nodom = " ".join(re.split(r' formerly: ', s_nodom))
        
    s_clean = PUNCT_REGEX.sub(' ', s_nodom)
    tokens = [w for w in s_clean.split() if len(w) >= 2 and w not in LEGAL_SUFFIXES and w not in COMMON_TOKENS]
    
    if tokens:
        compact = "".join(tokens)
    else:
        compact = re.sub(r'[^a-z0-9]', '', s_clean)
        
    return " ".join(tokens), set(tokens), compact

def clean_address(raw_addr: str):
    if not raw_addr or str(raw_addr).strip().lower() in ('none', 'null', 'nan'):
        return "", set(), set(), set(), set(), ""
    
    s_asc = anyascii.anyascii(str(raw_addr)).lower()
    raw_nums = re.findall(r'\d+', s_asc)
    # Normalize numbers, strip leading zeros
    numbers = set(n.lstrip('0') or '0' for n in raw_nums)
    
    s_clean = PUNCT_REGEX.sub(' ', s_asc)
    words = s_clean.split()
    tokens = [w for w in words if len(w) >= 2 and not w.isdigit()]
    sub_tokens = [w for w in tokens if len(w) >= 3 and w not in ADDR_STOP_WORDS and w not in COMMON_TOKENS]
    
    # Compact alphanumeric address for exact sole-occupant hashing
    a_compact = re.sub(r'[^a-z0-9]', '', s_asc)
    
    signatures = set()
    first_num = sorted(list(numbers))[0] if numbers else None
    if first_num and sub_tokens:
        for t in sub_tokens[:2]:
            signatures.add(f"{first_num}_{t}")
            
    return " ".join(tokens), set(tokens), set(sub_tokens), numbers, signatures, a_compact
