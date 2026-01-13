import joblib

MODEL_PATH = "models/pt_sentiment_model.pkl"

# Carrega uma única vez (cache em memória)
_bundle = joblib.load(MODEL_PATH)

_model = _bundle["model"]
_vectorizer = _bundle["vectorizer"]
_preprocess = _bundle["preprocess_fn"]

def predict(text, lang="pt"):
    clean = _preprocess(text)
    vec = _vectorizer.transform([clean])

    pred = _model.predict(vec)[0]
    proba = _model.predict_proba(vec).max()

    return pred, proba
