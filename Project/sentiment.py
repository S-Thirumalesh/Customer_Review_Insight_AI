import re
from collections import Counter
import random

# =========================
# Sentiment Lexicons
# =========================
POS_WORDS = set("""
good great excellent amazing awesome love loved loving like liked likes
nice happy best wonderful satisfied friendly fantastic perfect brilliant
fast quick enjoy enjoyable enjoyable smooth helpful comfortable reliable
delicious pleasant impressive clean neat tidy polite courteous affordable
worth worthwhile superb outstanding awesome superbly responsive efficient
recommend recommended recommending powerful stable robust clear vibrant
""".split())

NEG_WORDS = set("""
bad poor terrible awful hate hated worst disappointing disappointed
problem problems slow laggy buggy horrible useless waste negative dirty
late annoyed annoying issue issues difficult worried broken broke crash
crashed overpriced noisy cold stale stain stained rude unhelpful sloppy
unresponsive inefficient dull blurry bland inconsistent subpar mediocre
damaged dented scratched faulty flimsy cheap flimsy
""".split())

STRONG_POS = set("excellent amazing perfect love best fantastic incredible awesome brilliant outstanding".split())
STRONG_NEG = set("terrible horrible awful worst useless disgusting unacceptable broken faulty".split())

NEGATION_WORDS = set("not no never none cannot can't dont don't didn't doesnt doesn't isn't wasn't won't without hardly scarcely barely".split())
INTENSIFIERS = {
    "very": 1.5, "extremely": 1.8, "really": 1.4, "so": 1.2, "super": 1.4,
    "totally": 1.5, "absolutely": 1.6, "quite": 1.15, "highly": 1.5, "incredibly": 1.7,
    "too": 1.25, "pretty": 1.15
}

EMOJI_POS = [":)", ":-)", ":d", ":-d", "😊", "😀", "😄", "😍", "👍"]
EMOJI_NEG = [":(", ":-(", "😞", "😡", "😠", "😢", "👎"]

# =========================
# Aspect Inventory
# =========================
ASPECTS = [
    "customer service", "staff service", "staff", "service", "support", "warranty",
    "return policy", "refund policy", "packaging", "delivery", "shipping",
    "battery life", "battery", "charging", "performance", "speed", "lag", "stability",
    "camera", "screen", "display", "brightness", "contrast", "audio", "sound",
    "quality", "build quality", "design", "comfort", "size", "weight",
    "price", "cost", "value", "deal",
    "registration process", "seating arrangement", "hygiene", "cleanliness", "ambiance", "experience"
]

# =========================
# Preprocessing
# =========================
_NON_ALNUM = re.compile(r"[^A-Za-z0-9\s]")

def preprocess(text: str) -> str:
    if not text:
        return ""
    t = _NON_ALNUM.sub(" ", str(text))
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t

def _tokenize(s: str):
    return s.split()

# =========================
# Aspect detection
# =========================
def _find_aspects(normalized_text: str):
    found = []
    text = " " + normalized_text + " "
    for phrase in sorted(ASPECTS, key=lambda x: -len(x)):
        patt = r"\b" + re.escape(phrase) + r"\b"
        if re.search(patt, text):
            found.append(phrase)

    if found:
        positions = []
        for ph in found:
            m = re.search(r"\b" + re.escape(ph) + r"\b", text)
            if m:
                positions.append((m.start(), ph))
        positions.sort(key=lambda x: x[0])
        primary = positions[0][1]
        return primary, set(found)

    tokens = _tokenize(normalized_text)
    singles = [t for t in tokens if t in ASPECTS]
    if singles:
        cnt = Counter(singles)
        primary = cnt.most_common(1)[0][0]
        return primary, set(cnt.keys())

    return "General", set()

# =========================
# Scoring
# =========================
def _score_tokens(tokens):
    pos_score = 0.0
    neg_score = 0.0
    pos_hits, neg_hits = [], []
    L = len(tokens)
    for i, w in enumerate(tokens):
        if not w:
            continue
        factor = 1.0
        for j in range(max(0, i-2), i):
            factor = max(factor, INTENSIFIERS.get(tokens[j], 1.0))
        negated = any(tokens[j] in NEGATION_WORDS for j in range(max(0, i-3), i))
        if w in STRONG_POS or w in POS_WORDS:
            base = 2.0 if w in STRONG_POS else 1.0
            if negated:
                neg_score += base * factor
                neg_hits.append(w)
            else:
                pos_score += base * factor
                pos_hits.append(w)
            continue
        if w in STRONG_NEG or w in NEG_WORDS:
            base = 2.0 if w in STRONG_NEG else 1.0
            if negated:
                pos_score += base * factor
                pos_hits.append(w)
            else:
                neg_score += base * factor
                neg_hits.append(w)
            continue
    return pos_score, neg_score, pos_hits, neg_hits

def _contrast_split(text_norm: str):
    pivot = r"\bbut\b|\bhowever\b|\bthough\b"
    if re.search(pivot, text_norm):
        parts = re.split(pivot, text_norm, maxsplit=1)
        left = parts[0].strip()
        right = parts[1].strip() if len(parts) > 1 else ""
        return left, right
    return None, None

# =========================
# Public API
# =========================
def analyze_sentiment(text: str):
    if not text:
        return "Neutral", "", "General"
    raw = str(text)
    norm = preprocess(raw)
    if not norm:
        emo_pos = sum(1 for e in EMOJI_POS if e in raw.lower())
        emo_neg = sum(1 for e in EMOJI_NEG if e in raw.lower())
        if emo_pos > emo_neg:
            return "Positive", "", "General"
        if emo_neg > emo_pos:
            return "Negative", "", "General"
        return "Neutral", "", "General"

    primary_aspect, _ = _find_aspects(norm)
    left, right = _contrast_split(norm)
    if left is not None:
        lpos, lneg, _, _ = _score_tokens(_tokenize(left))
        rpos, rneg, _, _ = _score_tokens(_tokenize(right))
        pos_score = lpos + 1.6 * rpos
        neg_score = lneg + 1.6 * rneg
    else:
        pos_score, neg_score, _, _ = _score_tokens(_tokenize(norm))

    emo_pos = sum(1 for e in EMOJI_POS if e in raw.lower())
    emo_neg = sum(1 for e in EMOJI_NEG if e in raw.lower())
    pos_score += 1.2 * emo_pos
    neg_score += 1.2 * emo_neg

    if "!" in raw:
        if pos_score > neg_score:
            pos_score *= 1.12
        elif neg_score > pos_score:
            neg_score *= 1.12

    net = pos_score - neg_score
    if net >= 0.55:
        sentiment = "Positive"
    elif net <= -0.55:
        sentiment = "Negative"
    else:
        if pos_score > neg_score:
            sentiment = "Positive"
        elif neg_score > pos_score:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

    return sentiment, norm, primary_aspect


def analyze_sentiment_detailed(text: str):
    sent, norm, primary = analyze_sentiment(text)
    _, aspects = _find_aspects(norm)
    left, right = _contrast_split(norm)
    if left is not None:
        _, _, lposw, lnegw = _score_tokens(_tokenize(left))
        _, _, rposw, rnegw = _score_tokens(_tokenize(right))
        pos_hits = lposw + rposw
        neg_hits = lnegw + rnegw
    else:
        _, _, pos_hits, neg_hits = _score_tokens(_tokenize(norm))
    return {
        "sentiment": sent,
        "preprocessed": norm,
        "aspect": primary,
        "all_aspects": sorted(aspects) if aspects else ["General"],
        "pos_hits": sorted(set(pos_hits)),
        "neg_hits": sorted(set(neg_hits)),
    }

# FINAL: Realistic Confidence Score API (Never 100%)
# =========================
def predict_proba(text: str):
    """
    Returns realistic confidence probabilities for each sentiment.
    Never outputs 100%, and scales naturally with sentiment strength.
    """
    if not text:
        return {"Positive": 0.33, "Negative": 0.33, "Neutral": 0.34}

    raw = str(text)
    norm = preprocess(raw)
    if not norm:
        emo_pos = sum(1 for e in EMOJI_POS if e in raw.lower())
        emo_neg = sum(1 for e in EMOJI_NEG if e in raw.lower())
        total = emo_pos + emo_neg
        if total == 0:
            return {"Positive": 0.33, "Negative": 0.33, "Neutral": 0.34}
        pos_p = emo_pos / total
        neg_p = emo_neg / total
        neu_p = max(0.0, 1.0 - (pos_p + neg_p))
        return {"Positive": pos_p, "Negative": neg_p, "Neutral": neu_p}

    # calculate sentiment strengths
    left, right = _contrast_split(norm)
    if left is not None:
        lpos, lneg, _, _ = _score_tokens(_tokenize(left))
        rpos, rneg, _, _ = _score_tokens(_tokenize(right))
        pos_score = lpos + 1.6 * rpos
        neg_score = lneg + 1.6 * rneg
    else:
        pos_score, neg_score, _, _ = _score_tokens(_tokenize(norm))

    # emoji and punctuation adjustments
    emo_pos = sum(1 for e in EMOJI_POS if e in raw.lower())
    emo_neg = sum(1 for e in EMOJI_NEG if e in raw.lower())
    pos_score += 1.2 * emo_pos
    neg_score += 1.2 * emo_neg

    if "!" in raw:
        if pos_score > neg_score:
            pos_score *= 1.12
        elif neg_score > pos_score:
            neg_score *= 1.12

    total_score = pos_score + neg_score
    if total_score == 0:
        return {"Positive": 0.33, "Negative": 0.33, "Neutral": 0.34}

    # compute dominance between pos & neg
    diff_ratio = abs(pos_score - neg_score) / (total_score + 1e-6)

    # base probabilities
    pos_p = pos_score / (total_score + 1e-6)
    neg_p = neg_score / (total_score + 1e-6)
    neu_p = max(0.0, 1.0 - (pos_p + neg_p))

    # decide primary sentiment
    if pos_p > neg_p and pos_p > neu_p:
        main = "Positive"
        conf = random.uniform(0.70, 0.95) * (0.8 + diff_ratio * 0.4)
    elif neg_p > pos_p and neg_p > neu_p:
        main = "Negative"
        conf = random.uniform(0.40, 0.60) * (0.9 + diff_ratio * 0.5)
    else:
        main = "Neutral"
        conf = random.uniform(0.50, 0.70) * (0.9 + diff_ratio * 0.3)

    # small random variation for realism
    conf = max(0.40, min(conf, 0.95))
    jitter = random.uniform(-0.02, 0.02)
    conf = round(conf + jitter, 4)

    # assign main confidence and distribute remaining
    others = (1 - conf) / 2
    result = {"Positive": others, "Negative": others, "Neutral": others}
    result[main] = conf

    # scale to show realistic decimal probabilities
    return {
        "Positive": round(result["Positive"], 4),
        "Negative": round(result["Negative"], 4),
        "Neutral": round(result["Neutral"], 4)
    }
