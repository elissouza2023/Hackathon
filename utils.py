import joblib
import os
import re

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "modelo_lr_sentimentos.pkl")
VEC_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)

# ===== NLP (IGUAL AO NOTEBOOK) =====

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#", " ", text)
    text = re.sub(r"[^a-záàâãéèêíïóôõöúçñ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def handle_negations(text):
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
    intensifiers = r"(muito|bem|super|extremamente|bastante|totalmente)"
    text = re.sub(
        rf"\b{intensifiers}\s+(ruim|péssimo|horrível|insatisfeito|decepcionado|bom|ótimo|excelente|confuso)",
        r"\1_\2",
        text
    )
    return text

def handle_negative_events(text):
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

# ===== PREDIÇÃO =====

def predict(text, lang="pt"):
    clean = preprocess_text(text)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec).max()
    return pred, proba
