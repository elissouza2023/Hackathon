import joblib
import os
import re

BASE_DIR = os.path.dirname(__file__)

# ============================
# Load models
# ============================

pt_model = joblib.load(os.path.join(BASE_DIR, "modelo_lr_sentimentos.pkl"))
pt_tfidf = joblib.load(os.path.join(BASE_DIR, "tfidf.pkl"))

en_model = joblib.load(os.path.join(BASE_DIR, "en_model.pkl"))
en_tfidf = joblib.load(os.path.join(BASE_DIR, "en_tfidf.pkl"))

# ============================
# 🇧🇷 PORTUGUESE
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
        r"\b(problema)\s+(persistiu|continua)"
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

    for c in contractions:
        text = text.replace(c, contractions[c])

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
            result.append("NOT_" + token)
            negate -= 1
        else:
            result.append(token)

    return " ".join(result)

def handle_intensifiers_en(text):
    intensifiers = r"very|really|so|too|quite|extremely|absolutely|totally|highly|super"
    adjectives = r"bad|terrible|awful|poor|confusing|disappointed|good|great|excellent|amazing|perfect|awesome"
    pattern = rf"\b({intensifiers})\s+({adjectives})\b"
    return re.sub(pattern, r"\1_\2", text)

def handle_negative_events_en(text):
    patterns = [
        r"\b(delivery|shipping)\s+(delayed|late)",
        r"\b(box|package)\s+(damaged|broken|crushed)",
        r"\b(product|item)\s+(damaged|broken|defective)",
        r"\b(item)\s+arrived\s+(damaged|broken)",
        r"\b(problem|issue)\s+(persists|continues|remains)"
    ]
    for p in patterns:
        text = re.sub(p, lambda m: "_".join(m.group(0).split()), text)
    return text

def preprocess_en(text):
    text = clean_text_en(text)
    text = handle_negations_en(text)
    text = handle_intensifiers_en(text)
    text = handle_negative_events_en(text)
    return text

# ============================
# 🔮 Prediction
# ============================

def predict(text, lang="pt"):
    if lang == "pt":
        clean = preprocess_pt(text)
        vec = pt_tfidf.transform([clean])
        pred = pt_model.predict(vec)[0]
        proba = pt_model.predict_proba(vec).max()

    elif lang == "en":
        clean = preprocess_en(text)
        vec = en_tfidf.transform([clean])
        pred = en_model.predict(vec)[0]
        proba = en_model.predict_proba(vec).max()

    else:
        raise ValueError("Language not supported")

    return pred, proba
