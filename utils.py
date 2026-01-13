import joblib
import re

MODELS = {
    "pt": {
        "model": "modelo_lr_sentimentos.pkl",
        "vectorizer": "tfidf.pkl"
    },
    "en": {
        "model": "modelo_lr_sentimentos_en.pkl",
        "vectorizer": "tfidf-en.pkl"
    },
    "es": {
        "model": "modelo_lr_sentimentos_es.pkl",
        "vectorizer": "tfidf_es.pkl"
    }
}

# === PIPELINE DE NLP (IGUAL AO NOTEBOOK) ===

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#", " ", text)
    text = re.sub(r"[^a-záàâãéèêíïóôõöúçñ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def handle_negations(text):
    if not isinstance(text, str):
        return ""
    patterns = [
        r"\bnão\s+(\w+)",
        r"\bnao\s+(\w+)",
        r"\bnunca\s+(\w+)",
        r"\bjamais\s+(\w+)"
    ]
    for p in patterns:
        text = re.sub(p, r"nao_\1", text)
    return text

def handle_intensifiers(text):
    if not isinstance(text, str):
        return ""
    intensifiers = r"(muito|bem|super|extremamente|bastante|totalmente)"
    text = re.sub(
        rf"\b{intensifiers}\s+(ruim|péssimo|horrível|insatisfeito|decepcionado|bom|ótimo|excelente|confuso)",
        r"\1_\2",
        text
    )
    return text

def handle_negative_events(text):
    if not isinstance(text, str):
        return ""
    patterns = [
        r"\b(entrega)\s+(atrasou|demorou)",
        r"\b(caixa|produto)\s+(amassada|quebrada|danificada)",
        r"\b(problema)\s+(persistiu|continua)",
    ]
    for p in patterns:
        text = re.sub(p, r"\1_\2", text)
    return text

def preprocess_text(text):
    text = clean_text(text)
    text = handle_negations(text)
    text = handle_intensifiers(text)
    text = handle_negative_events(text)
    return text

# === MODELO ===

def load_model(lang):
    paths = MODELS[lang]
    model = joblib.load(paths["model"])
    vectorizer = joblib.load(paths["vectorizer"])
    return model, vectorizer

def predict(text, lang):
    model, vectorizer = load_model(lang)

    clean = preprocess_text(text)   # ⭐ agora igual ao treino
    X = vectorizer.transform([clean])

    pred = model.predict(X)[0]
    proba = model.predict_proba(X).max()

    return pred, proba
