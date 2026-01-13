import streamlit as st
from utils import predict

st.set_page_config(page_title="Kaida Search", page_icon="🌱", layout="centered")

st.title("🌱 Kaida Search")
st.write("Análise de Sentimentos Multilíngue para ESG avaliações de clientes")

lang = st.selectbox("Idioma do texto:", ["pt", "en", "es"])
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
