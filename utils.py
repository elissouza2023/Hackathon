import joblib
import os

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "pt_sentiment_model.pkl")

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
