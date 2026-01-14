import streamlit as st
from utils import predict
import pandas as pd


st.set_page_config(page_title="Análise de Sentimentos", page_icon="🌱", layout="centered")

st.title("🌱 Análise de Sentimentos")
st.write("Análise de Sentimentos Multilíngue para categorizar as avaliações dos clientes em : Positivo, Negativo ou Neutro")

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
st.write("### Ou envie um arquivo CSV")

uploaded_file = st.file_uploader(
    "O CSV deve conter uma coluna chamada 'text'",
    type=["csv"]
)

# Mapa entre interface e modelo
lang_map = {
    "Português - BR": "pt",
    "English - US": "en",
    "Spanish - 419": "es"
}

# Mapeamento de classes por idioma
CLASS_MAPPING = {
    "pt": {
        "Positivo": "Positivo",
        "Negativo": "Negativo",
        "Neutro": "Neutro"
    },
    "es": {
        "Positivo": "Positivo",
        "Negativo": "Negativo",
        "Neutral": "Neutro"
    },
    "en": {
        "Positive": "Positivo",
        "Negative": "Negativo",
        "Neutral": "Neutro"
    }
}

use_file = uploaded_file is not None


if st.button("Analisar"):
    if not use_file and text.strip() == "":
        st.warning("Digite um texto ou envie um CSV.")
    
    else:
        if use_file:
            df = pd.read_csv(uploaded_file)

            df.columns = df.columns.str.strip().str.lower()
            if "text" not in df.columns:
                st.error("O CSV deve conter uma coluna chamada 'text'.")
            else:
                results = []

                for t in df["text"]:
                    raw_label, prob = predict(str(t), lang)
                    raw_label = raw_label.strip()
                    label = CLASS_MAPPING[lang].get(raw_label, f"Classe desconhecida: {raw_label}")

                    results.append((t, label, prob))

                result_df = pd.DataFrame(results, columns=["Texto", "Sentimento", "Confiança"])
                st.dataframe(result_df)

                st.download_button(
                    "Baixar resultados",
                    result_df.to_csv(index=False).encode("utf-8"),
                    file_name="resultado_sentimentos.csv",
                    mime="text/csv"
                )

        else:
            raw_label, prob = predict(text, lang)
            raw_label = raw_label.strip()
            label = CLASS_MAPPING[lang].get(raw_label, f"Classe desconhecida: {raw_label}")

            if label == "Positivo":
                st.success("Sentimento POSITIVO")
            elif label == "Negativo":
                st.error("Sentimento NEGATIVO")
            elif label == "Neutro":
                st.info("Sentimento NEUTRO")
            else:
                st.warning(label)

            st.write(f"Confiança do modelo: {prob:.2%}")

