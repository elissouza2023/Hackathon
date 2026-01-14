import joblib
import os
import re

BASE_DIR = os.path.dirname(__file__)

# ============================
# Carregamento dos modelos
# ============================

MODELS = {
    "pt": {
        "model": joblib.load(os.path.join(BASE_DIR, "modelo_lr_sentimentos.pkl")),
        "tfidf": joblib.load(os.path.join(BASE_DIR, "tfidf.pkl"))
    },
    "en": {
        "model": joblib.load(os.path.join(BASE_DIR, "en_model.pkl")),
        "tfidf": joblib.load(os.path.join(BASE_DIR, "en_tfidf.pkl"))
    }
}

# ============================
# 🇧🇷 PORTUGUÊS
# ============================

def clean_text_pt(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#", " ", text)
    text = re.sub(r"[^a-záàâãéèêíïóôõöúçñ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def handle_negations_pt(text):
    patterns = [
        r"\bnão\s+(\w+)",
        r"\bnao\s+(\w+)",
        r"\bnunca\s+(\w+)",
        r"\bjamais\s+(\w+)"
    ]
    for p in patterns:
        text = re.sub(p, r"nao_\1", text)
    return text

def handle_intensifiers_pt(text):
    intensifiers = r"(muito|bem|super|extremamente|bastante|totalmente)"
    text = re.sub(
        rf"\b{intensifiers}\s+(ruim|péssimo|horrível|insatisfeito|decepcionado|bom|ótimo|excelente|confuso)",
        r"\1_\2",
        text
    )
    return text

def handle_negative_events_pt(text):
    patterns = [
        r"\b(entrega)\s+(atrasou|demorou)",
        r"\b(caixa|produto)\s+(amassada|quebrada|danificada)",
        r"\b(problema)\s+(persistiu|continua)",
    ]
    for p in patterns:
        text = re.sub(p, r"\1_\2", text)
    return text

def preprocess_pt(text):
    text = clean_text_pt(text)
    text = handle_negations_pt(text)
    text = handle_intensifiers_pt(text)
    text = handle_negative_events_pt(text)
    return text

# ============================
# 🇺🇸 ENGLISH
# ============================

def clean_text_en(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+", " ", text)

    contractions = {
        "can't": "can not", "won't": "will not", "don't": "do not",
        "doesn't": "does not", "didn't": "did not", "isn't": "is not",
        "aren't": "are not", "wasn't": "was not", "weren't": "were not",
        "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
        "couldn't": "could not", "shouldn't": "should not", "wouldn't": "would not",
        "n't": " not"
    }

    for c, e in contractions.items():
        text = text.replace(c, e)

    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def handle_negations_en(text, window=2):
    tokens = text.split()
    negations = {"not", "no", "never"}

    result = []
    negate = 0

    for token in tokens:
        if token in negations:
            negate = window
            result.append(token)
            continue

        if negate > 0:
            result.append(f"NOT_{token}")
            negate -= 1
        else:
            result.append(token)

    return " ".join(result)

def handle_intensifiers_en(text):
    intensifiers = r"very|really|so|too|quite|extremely|absolutely|totally|highly|super"
    adjectives = r"bad|terrible|awful|poor|confusing|disappointed|good|great|excellent|amazing|perfect|awesome"
    pattern = rf"\b({intensifiers})\s+({adjectives})\b"
    return re.sub(pattern, r"\1_\2", text)

def handle_negative_

