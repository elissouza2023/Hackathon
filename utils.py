import os
import re
import unicodedata
import joblib

# Diretório base do módulo (onde estão os arquivos .pkl)
BASE_DIR = os.path.dirname(__file__)

# ===============================
# Carregamento dos modelos e vectorizers
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
# NORMALIZAÇÃO UNIVERSAL (comum a todos os idiomas)
# =====================================================
def normalize_text(text):
    text = str(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    return text

# =====================================================
# PRÉ-PROCESSAMENTO ESPECÍFICO POR IDIOMA
# =====================================================
def preprocess_pt(text):
    text = normalize_text(text)
    text = re.sub(r"http\S+|www\S+|@\w+|#", " ", text)
    text = re.sub(r"[^a-záàâãéèêíïóôõöúçñ\s]", " ", text)
    
    # Preservar negação + adjetivo (ex: "não bom" → "nao_bom")
    text = re.sub(r"\b(não|nao|nunca|jamais)\s+(\w+)", r"nao_\2", text)
    
    # Intensificadores + adjetivos (ex: "muito bom" → "muito_bom")
    text = re.sub(
        r"\b(muito|bem|super|extremamente|bastante|totalmente)\s+(bom|ótimo|excelente|ruim|péssimo|horrível)",
        r"\1_\2",
        text,
    )
    return re.sub(r"\s+", " ", text).strip()


def preprocess_en(text):
    text = normalize_text(text)
    text = re.sub(r"http\S+|www\S+|@\w+", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)  # reduz "goooood" → "good"
    text = re.sub(r"[^a-z\s']", " ", text)
    
    text = re.sub(r"\b(not|no|never)\s+(\w+)", r"not_\2", text)
    text = re.sub(
        r"\b(very|really|so|too|extremely|super)\s+(good|great|excellent|bad|terrible|awful)",
        r"\1_\2",
        text,
    )
    return re.sub(r"\s+", " ", text).strip()


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


# Mapa de funções de pré-processamento
PREPROCESSORS = {
    "pt": preprocess_pt,
    "en": preprocess_en,
    "es": preprocess_es,
}

# =====================================================
# FUNÇÃO PRINCIPAL DE PREDIÇÃO
# =====================================================
def predict(text, lang="pt"):
    """
    Realiza a predição de sentimento para um texto dado o idioma.
    
    Args:
        text (str): Texto a ser analisado
        lang (str): Código do idioma ('pt', 'en' ou 'es')
    
    Returns:
        tuple: (label: str, probabilidade: float)
    """
    if lang not in MODELS:
        raise ValueError(f"Idioma não suportado: {lang}. Use 'pt', 'en' ou 'es'.")

    bundle = MODELS[lang]
    preprocess = PREPROCESSORS[lang]

    # Pré-processamento
    clean = preprocess(text)

    # Tratamento de textos muito curtos ou vazios
    if not clean.strip():
        return "Neutro", 0.50

    # Aqui você pode adicionar regras para textos muito curtos (opcional, mas recomendado)
    # Exemplo:
    # if len(clean.split()) <= 2:
    #     palavras_pos = {"gostei", "adorei", "amei", "bom", "ótimo"}
    #     palavras_neg = {"odiei", "ruim", "péssimo", "horrível"}
    #     palavras = set(clean.split())
    #     if palavras & palavras_pos:
    #         return "Positivo", 0.85
    #     if palavras & palavras_neg:
    #         return "Negativo", 0.85
    #     return "Neutro", 0.60

    # Vetorização e predição
    vec = bundle["vectorizer"].transform([clean])
    pred = bundle["model"].predict(vec)[0]
    proba = bundle["model"].predict_proba(vec).max()

    return pred, proba
