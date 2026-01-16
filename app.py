import streamlit as st
from utils import predict
import pandas as pd
import streamlit as st
import base64


def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


set_background("fundo.jpg")

st.markdown("""
<style>
/* Texto geral */
html, body, [class*="css"]  {
    color: #ffffff !important;
}

/* Títulos */
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
}

/* Labels de inputs */
label {
    color: #ffffff !important;
}

/* Texto dentro dos inputs */
input, textarea {
    color: #ffffff !important;
}

/* Botões */
button {
    color: #3c3c3c !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.stApp::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.4);
    z-index: -1;
}
</style>
""", unsafe_allow_html=True)



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
            df = pd.read_csv(uploaded_file, encoding="utf-8-sig")

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

st.markdown("""
<style>
.footer {
    position: fixed;
    bottom: 10px;
    left: 0;
    width: 100%;
    text-align: center;
    color: #ffffff;
    font-size: 12px;
    opacity: 0.8;
}
</style>

<div class="footer">
© 2026 •  Análise de Sentimentos • Todos os Direitos Reservados
</div>
""", unsafe_allow_html=True)
