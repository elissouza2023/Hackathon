import streamlit as st
from utils import predict

st.set_page_config(page_title="Anáiise de Sentimentos", page_icon="🌱", layout="centered")

st.title("🌱 Análise de Sentimentos")
st.write("Análise de Sentimentos Multilíngue para avaliações de clientes")

lang = st.selectbox("Idioma do texto:", ["Português - BR", "English - US", "Spanish - 419"])
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
