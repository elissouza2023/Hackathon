import streamlit as st
from utils import predict

st.set_page_config(page_title="Análise de Sentimentos", page_icon="🌱", layout="centered")

st.title("🌱 Análise de Sentimentos")
st.write("Análise de Sentimentos Multilíngue para avaliações de clientes")

lang_ui = st.selectbox(
    "Idioma do texto:",
    ["Português - BR", "English - US", "Spanish - 419"]
)

# Mapa entre interface e modelo
lang_map = {
    "Português - BR": "pt",
    "English - US": "en",
    "Spanish - 419": "es"
}

lang = lang_map[lang_ui]

text = st.text_area("Digite o texto para análise:")

if st.button("Analisar"):
    if text.strip() == "":
        st.warning("Digite um texto.")
    else:
        label, prob = predict(text, lang)

        if label == 1:
            st.success("Sentimento POSITIVO")
        else:
            st.error("Sentimento NEGATIVO")

        st.write(f"Confiança do modelo: {prob:.2%}")

