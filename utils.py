import os
import joblib
import re
import unicodedata

BASE_DIR = os.path.dirname(__file__)

# ===============================
# Load models
# ===============================
MODELS = {
    "pt": {
        "model": joblib.load(os.path.join(BASE_DIR, "modelo_lr_sentimentos.pkl")),
        "vectorizer": joblib.load(os.path.join(BASE_DIR, "tfidf.pkl")),
    },
    "en": {
        "model": joblib.load(os.path.join(BASE_DIR, "en_model.pkl")),
        "vectorizer": joblib.load(os.path.join(BASE_DIR, "en_tfidf.pkl")),
    },
    "es": {
        "model": joblib.load(os.path.join(BASE_DIR, "es_model.pkl")),
        "vectorizer": joblib.load(os.path.join(BASE_DIR, "es_tfidf.pkl")),
    },
}

# =====================================================
# NORMALIZAÇÃO UNIVERSAL
# =====================================================

def normalize_text(text):
    text = str(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    return text


# =====================================================
# PORTUGUESE
# =====================================================

def preprocess_pt(text):

    text = normalize_text(text)

    text = re.sub(r"http\S+|www\S+|@\w+|#", " ", text)

    text = re.sub(r"[^a-záàâãéèêíïóôõöúçñ\s]", " ", text)

    text = re.sub(r"\b(não|nao|nunca|jamais)\s+(\w+)", r"nao_\2", text)

    text = re.sub(
        r"\b(muito|bem|super|extremamente|bastante|totalmente)\s+(bom|ótimo|excelente|ruim|péssimo|horrível)",
        r"\1_\2",
        text,
    )

    return re.sub(r"\s+", " ", text).strip()


# =====================================================
# ENGLISH
# =====================================================

def preprocess_en(text):

    text = normalize_text(text)

    text = re.sub(r"http\S+|www\S+|@\w+", " ", text)

    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    text = re.sub(r"[^a-z\s']", " ", text)

    text = re.sub(r"\b(not|no|never)\s+(\w+)", r"not_\2", text)

    text = re.sub(
        r"\b(very|really|so|too|extremely|super)\s+(good|great|excellent|bad|terrible|awful)",
        r"\1_\2",
        text,
    )

    return re.sub(r"\s+", " ", text).strip()


# =====================================================
# SPANISH
# =====================================================

def preprocess_es(text):

    text = normalize_text(text)

    text = re.sub(r"http\S+|www\S+|@\w+", " ", text)

    text = re.sub(r"[^a-záéíóúüñ\s]", " ", text)

    text = re.sub(r"\b(no|nunca|jamás|jamas|tampoco)\s+(\w+)", r"no_\2", text)

    text = re.sub(
        r"\b(muy|super|extremadamente|bastante|totalmente)\s+(bueno|excelente|malo|terrible|horrible)",
        r"\1_\2",
        text,
    )

    return re.sub(r"\s+", " ", text).strip()


# =====================================================
# MAPA DE PREPROCESSADORES
# =====================================================

PREPROCESSORS = {
    "pt": preprocess_pt,
    "en": preprocess_en,
    "es": preprocess_es,
}


# =====================================================
# FUNÇÃO DE PREDIÇÃO
# =====================================================

def predict(text, lang="pt"):

    bundle = MODELS[lang]

    preprocess = PREPROCESSORS[lang]

    clean = preprocess(text)

    # proteção contra texto vazio
    if clean == "":
        return "Neutro", 0.50

    vec = bundle["vectorizer"].transform([clean])

    pred = bundle["model"].predict(vec)[0]

    proba = bundle["model"].predict_proba(vec).max()

    return pred, proba
    clean = preprocess(text)
    vec = bundle["vectorizer"].transform([clean])
    pred = bundle["model"].predict(vec)[0]
    proba = bundle["model"].predict_proba(vec).max()

    return pred, proba
