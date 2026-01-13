import joblib

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

def load_model(lang):
    paths = MODELS[lang]
    model = joblib.load(paths["model"])
    vectorizer = joblib.load(paths["vectorizer"])
    return model, vectorizer

def predict(text, lang):
    model, vectorizer = load_model(lang)
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X).max()
    return pred, proba
