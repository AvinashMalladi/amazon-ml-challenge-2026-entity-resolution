import re
import anyascii

LEGAL_SUFFIXES = {
    # English / Global
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'llp', 'ltd', 'limited', 
    'pvt', 'private', 'co', 'company', 'services', 'enterprises', 'solutions', 'holdings',
    'group', 'technologies', 'technology', 'tech', 'ventures', 'international', 'intl',
    'management', 'consulting', 'consultancy', 'consultants', 'associates', 'assoc',
    'center', 'centre', 'industries', 'global', 'systems', 'system',
    # French
    'sarl', 'sasu', 'sas', 'eurl', 'sci', 'sa', 'snc', 'gie', 'selarl', 'scop', 'sem',
    'association', 'asso', 'societe', 'ste', 'etablissement', 'ets', 'cie', 'freres', 'fils'
}

ADDR_STOP_WORDS = {
    # English / Global
    'road', 'street', 'avenue', 'drive', 'lane', 'place', 'boulevard', 'court', 'way',
    'circle', 'highway', 'parkway', 'suite', 'floor', 'block', 'plot', 'door', 'flat',
    'house', 'building', 'tower', 'complex', 'near', 'opp', 'opposite', 'behind', 'beside',
    'rd', 'st', 'ave', 'dr', 'ln', 'pl', 'blvd', 'ct', 'cir', 'hwy', 'pkwy', 'ste', 'fl',
    'north', 'south', 'east', 'west', 'po', 'box', 'pmb',
    # French
    'rue', 'avenue', 'boulevard', 'allee', 'impasse', 'chemin', 'cours', 'place', 'route',
    'quai', 'passage', 'cite', 'square', 'residence', 'immeuble', 'batiment', 'etage',
    'r', 'av', 'bd', 'all', 'imp', 'chm', 'rte', 'pl', 'res', 'bat', 'bis', 'ter'
}

COMMON_TOKENS = {
    'the', 'and', 'of', 'in', 'for', 'at', 'on', 'to', 'a', 'an', 'de', 'du', 'des', 'la', 'le', 'les', 'et'
}

DOMAIN_REGEX = re.compile(r'\.(com|org|net|in|co|fr|gov|edu|biz|info|io)(\.[a-z]{2})?$', re.IGNORECASE)
PUNCT_REGEX = re.compile(r'[\.\-\_\,\/\:\;\(\)\[\]\<\>\&\"\'\`\+\#\@\*\=\\\~]')
NUM_REGEX = re.compile(r'\b\d+\b')

def clean_name(raw_name: str):
    if not raw_name:
        return "", set(), ""
    
    s_asc = anyascii.anyascii(str(raw_name)).lower()
    
    # Strip domain extensions
    words = s_asc.split()
    cleaned_words = [DOMAIN_REGEX.sub('', w) for w in words]
    s_nodom = " ".join(cleaned_words)
    
    s_clean = PUNCT_REGEX.sub(' ', s_nodom)
    tokens = [w for w in s_clean.split() if len(w) >= 2 and w not in LEGAL_SUFFIXES and w not in COMMON_TOKENS]
    
    compact = re.sub(r'[^a-z0-9]', '', s_clean)
    return " ".join(tokens), set(tokens), compact

def clean_address(raw_addr: str):
    if not raw_addr:
        return "", set(), set(), set()
    
    s_asc = anyascii.anyascii(str(raw_addr)).lower()
    numbers = set(NUM_REGEX.findall(s_asc))
    
    s_clean = PUNCT_REGEX.sub(' ', s_asc)
    tokens = [w for w in s_clean.split() if len(w) >= 3 and w not in ADDR_STOP_WORDS and w not in COMMON_TOKENS]
    
    signatures = set()
    for n in numbers:
        for t in tokens:
            if len(t) >= 4 and not t.isdigit():
                signatures.add(f"{n}_{t[:4]}")
                
    return " ".join(tokens), set(tokens), numbers, signatures
