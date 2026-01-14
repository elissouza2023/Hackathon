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

def han
