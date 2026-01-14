import os
import joblib
import re

BASE_DIR = os.path.dirname(__file__)

# ===============================
# Load models
# ===============================
MODELS = {
    "pt": {
        "model": joblib.load(os.path.join(BASE_DIR, "models/modelo_lr_sentimentos.pkl")),
        "vectorizer": joblib.load(os.path.join(BASE_DIR, "models/tfidf.pkl")),
    },
    "en": {
        "model": joblib.load(os.path.join(BASE_DIR, "models/en_model.pkl")),
        "vectorizer": joblib.load(os.path.join(BASE_DIR, "models/en_tfidf.pkl")),
    },
    "es": {
        "model": joblib.load(os.path.join(BASE_DIR, "models/es_model.pkl")),
        "vectorizer": joblib.load(os.path.join(BASE_DIR, "models/es_tfidf.pkl")),
    },
}

# =====================================================
# ===============  PORTUGUESE  ========================
# =====================================================

def clean_text_pt(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#", " ", text)
    text = re.sub(r"[^a-záàâãéèêíïóôõöúçñ\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def handle_negations_pt(text):
    return re.sub(r"\b(não|nao|nunca|jamais)\s+(\w+)", r"nao_\2", text)

def handle_intensifiers_pt(text):
    return re.sub(r"\b(muito|bem|super|extremamente|bastante|totalmente)\s+(bom|ótimo|excelente|ruim|péssimo|horrível)",
                  r"\1_\2", text)

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


# =====================================================
# ===============  ENGLISH  ===========================
# =====================================================

def clean_text_en(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def handle_negations_en(text):
    return re.sub(r"\b(not|no|never)\s+(\w+)", r"not_\2", text)

def handle_intensifiers_en(text):
    return re.sub(r"\b(very|really|so|too|extremely|super)\s+(good|great|excellent|bad|terrible|awful)",
                  r"\1_\2", text)

def handle_negative_events_en(text):
    patterns = [
        r"\b(delivery|shipping)\s+(delayed|late)",
        r"\b(box|product)\s+(broken|damaged)",
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


# =====================================================
# ===============  SPANISH  ===========================
# =====================================================

def clean_text_es(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+", " ", text)
    text = re.sub(r"[^a-záéíóúüñ\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def handle_negations_es(text):
    return re.sub(r"\b(no|nunca|jamás|jamas|tampoco)\s+(\w+)", r"no_\2", text)

def handle_intensifiers_es(text):
    return re.sub(r"\b(muy|super|extremadamente|bastante|totalmente)\s+(bueno|excelente|malo|terrible|horrible)",
                  r"\1_\2", text)

def handle_negative_events_es(text):
    patterns = [
        r"\b(entrega|envío)\s+(retrasada|retrasado|tardía|tardío)",
        r"\b(caja|producto|paquete)\s+(abollada|roto|rota|dañado|dañada)",
    ]
    for p in patterns:
        text = re.sub(p, r"\1_\2", text)
    return text

def preprocess_es(text):
    text = clean_text_es(text)
    text = handle_negations_es(text)
    text = handle_intensifiers_es(text)
    text = handle_negative_events_es(text)
    return text


# =====================================================
# ===============  PREDICTOR  =========================
# =====================================================

PREPROCESSORS = {
    "pt": preprocess_pt,
    "en": preprocess_en,
    "es": preprocess_es
}

def predict(text, lang="pt"):
    bundle = MODELS[lang]
    preprocess = PREPROCESSORS[lang]

    clean = preprocess(text)
    vec = bundle["vectorizer"].transform([clean])
    pred = bundle["model"].predict(vec)[0]
    proba = bundle["model"].predict_proba(vec).max()

    return pred, proba
