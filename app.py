import streamlit as st
import pandas as pd
import base64
from utils import predict

# =====================================================
# CONFIGURAÇÃO DA PÁGINA
# =====================================================

st.set_page_config(
    page_title="Análise de Sentimentos",
    page_icon="🌱",
    layout="centered"
)

# =====================================================
# FUNDO COM IMAGEM
# =====================================================

def set_background(image_file: str):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("fundo.jpg")

# =====================================================
# CSS PERSONALIZADO
# =====================================================

st.markdown(
"""
<style>

.main > div {
    background-color: rgba(0,0,0,0.4);
    padding: 2rem;
    border-radius: 15px;
}

h1, h2, h3, .stMarkdown {
    color: white !important;
}

div.stButton > button {
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
    width: 100%;
    font-size: 1.2rem;
}

.footer {
    text-align: center;
    color: white;
    margin-top: 40px;
}

</style>
""",
unsafe_allow_html=True
)

# =====================================================
# INTERFACE
# =====================================================

st.title("🌱 Análise de Sentimentos")

st.markdown(
"Análise de Sentimentos Multilíngue para classificar avaliações em **Positivo, Negativo ou Neutro**."
)

lang_ui = st.selectbox(
"Idioma do texto:",
["Português - BR", "English - US", "Spanish - 419"]
)

LANG_MAP = {
"Português - BR": "pt",
"English - US": "en",
"Spanish - 419": "es"
}

lang = LANG_MAP[lang_ui]

text = st.text_area("Digite o texto para análise:")

uploaded_file = st.file_uploader(
"Ou envie um arquivo CSV com coluna chamada 'text'",
type=["csv"]
)

# =====================================================
# MAPEAMENTO DE CLASSES
# =====================================================

CLASS_MAPPING = {
"pt": {"Positivo":"Positivo","Negativo":"Negativo","Neutro":"Neutro"},
"es": {"Positivo":"Positivo","Negativo":"Negativo","Neutral":"Neutro"},
"en": {"Positive":"Positivo","Negative":"Negativo","Neutral":"Neutro"}
}

# =====================================================
# BOTÃO ANALISAR
# =====================================================

if st.button("Analisar"):

    # =============================
    # TEXTO DIGITADO
    # =============================

    if uploaded_file is None:

        if not text.strip():
            st.warning("Digite um texto.")
            st.stop()

        raw_label, prob = predict("" if pd.isna(text) else str(text), lang)

        label = CLASS_MAPPING[lang].get(raw_label.strip(), raw_label)

        st.markdown(
        f"""
        <div style="background-color:rgba(0,0,0,0.6);
                    padding:30px;
                    border-radius:15px;
                    text-align:center">

        <h2 style="color:white">Resultado</h2>

        <h3 style="color:{'#4CAF50' if label=='Positivo' else '#f44336' if label=='Negativo' else '#ff9800'}">
        {label}
        </h3>

        <h4 style="color:white">
        Confiança: {prob:.2%}
        </h4>

        </div>
        """,
        unsafe_allow_html=True
        )

    # =============================
    # CSV
    # =============================

    else:

        df = pd.read_csv(uploaded_file, encoding="utf-8-sig")

        df.columns = df.columns.str.strip().str.lower()

        if "text" not in df.columns:
            st.error("O CSV precisa ter coluna chamada 'text'")
            st.stop()

        results = []

        for t in df["text"]:

            raw_label, prob = predict("" if pd.isna(t) else str(t), lang)

            label = CLASS_MAPPING[lang].get(raw_label.strip(), raw_label)

            results.append((t, label, f"{prob:.2%}"))

        result_df = pd.DataFrame(
            results,
            columns=["Texto","Sentimento","Confiança"]
        )

        st.success("Análise concluída!")

        st.dataframe(result_df)

        csv = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
        "Baixar resultado CSV",
        csv,
        "resultado_sentimentos.csv",
        "text/csv"
        )

# =====================================================
# FOOTER
# =====================================================

st.markdown(
'<div class="footer">© 2026 • Análise de Sentimentos</div>',
unsafe_allow_html=True
)
