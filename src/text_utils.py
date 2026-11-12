"""Language detection, tokenisation, sentence splitting and query-intent helpers
for English and Hindi (Devanagari) text."""
from __future__ import annotations

import re

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")
TOKEN_RE = re.compile(r"[ऀ-ॿ]+|[A-Za-z]+|\d+(?:[.,]\d+)*")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?।])\s+|\n+")

# Small stop-word lists so that BM25 focuses on content words.
STOPWORDS = {
    # English
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "and", "or", "in", "on",
    "for", "with", "by", "at", "from", "as", "it", "this", "that", "what", "which", "who",
    "how", "can", "i", "my", "me", "do", "does", "under", "about", "any", "if", "per",
    "there", "their", "they", "them", "will", "would", "should", "much", "many", "get",
    # Hindi
    "का", "की", "के", "को", "में", "से", "है", "हैं", "और", "या", "पर", "यह", "वह", "क्या",
    "कौन", "कैसे", "लिए", "एक", "भी", "तो", "ही", "था", "थी", "थे", "मैं", "मुझे", "मेरा",
    "मेरी", "हो", "होता", "होती", "होते", "जो", "इस", "उस", "कर", "करें", "करना", "किस",
    "कितना", "कितनी", "कितने", "मिलता", "मिलती", "मिलते", "सकता", "सकती", "सकते",
}

# Words that signal what *kind* of information the user wants -> boost matching sections.
INTENT_KEYWORDS = {
    "eligibility": [
        "eligible", "eligibility", "who can", "qualify", "criteria", "entitled",
        "पात्र", "पात्रता", "कौन", "योग्य", "हकदार", "पात्रता", "मानदंड",
        "not eligible", "excluded", "exclusion", "अपात्र", "अपात्रता",
        "can i join", "can a ", "can an ", "join", "allowed", "age limit", "who gets",
        "शामिल", "ले सकता", "ले सकती", "जुड़ सकता", "जुड़ सकती", "उम्र", "आयु",
    ],
    "benefits": [
        "benefit", "amount", "how much", "money", "cover", "pension", "loan", "interest",
        "premium", "instalment", "installment", "how many", "days", "subsidy", "rate",
        "रुपये", "लाभ", "राशि", "कितना", "कितनी", "कितने", "सब्सिडी", "दिन",
        "पेंशन", "ऋण", "ब्याज", "प्रीमियम", "किस्त", "सहायता", "कवर",
    ],
    "documents": [
        "document", "documents", "papers", "proof", "kyc", "aadhaar",
        "दस्तावेज", "कागजात", "प्रमाण", "आधार",
    ],
    "apply": [
        "apply", "application", "register", "registration", "enrol", "enroll", "how to",
        "where to", "आवेदन", "पंजीकरण", "कैसे", "कहां", "कहाँ",
    ],
}

# Section headings (lower-case, both languages) that belong to each intent.
INTENT_SECTIONS = {
    "eligibility": ["eligibility", "exclusions", "पात्रता", "अपात्रता"],
    "benefits": ["benefits", "loan categories", "contribution", "deposit rules",
                 "withdrawal", "लाभ", "ऋण श्रेणियां", "अंशदान", "जमा", "निकासी"],
    "documents": ["documents required", "आवश्यक दस्तावेज"],
    "apply": ["how to apply", "helpline", "आवेदन कैसे करें", "हेल्पलाइन"],
}


def detect_language(text: str) -> str:
    """Return 'hi' if the text is mostly Devanagari, else 'en'."""
    dev = len(DEVANAGARI.findall(text))
    lat = len(LATIN.findall(text))
    if dev == 0 and lat == 0:
        return "en"
    return "hi" if dev >= lat else "en"


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens for English + Devanagari, stop-words removed."""
    toks = [t.lower() for t in TOKEN_RE.findall(text)]
    return [t for t in toks if t not in STOPWORDS and len(t) > 1 or t.isdigit()]


def split_sentences(text: str) -> list[str]:
    parts = [p.strip(" -•\t") for p in SENT_SPLIT_RE.split(text)]
    return [p for p in parts if len(p) > 3]


def detect_intents(query: str) -> list[str]:
    """Return the information needs in the query. Eligibility questions often also
    contain benefit words ("pension", "loan", "लाभ"), so eligibility takes priority."""
    q = f" {query.lower()} "
    found = [intent for intent, kws in INTENT_KEYWORDS.items() if any(k in q for k in kws)]
    return ["eligibility"] if "eligibility" in found else found
