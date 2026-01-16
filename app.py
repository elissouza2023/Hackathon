import streamlit as st
import pandas as pd
import base64
from utils import predict


# =========================
# FUNÇÃO DE FUNDO
# =========================
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


# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================
st.set_page_config(
    page_title="Análise de Sentimentos",
    page_icon="🌱",
    layout="centered"
)

set_background("fundo.jpg")


# =========================
# CSS GLOBAL
# =========================
st.markdown("""
<style>
/* Texto geral */
html, body, [class*="css"] {
    color: #ffffff !important;
}
/* Card de resultado */
.result-card,
.result-card * {
    color: #000000 !important;
}

/* Títulos */
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
}

/* Labels */
label {
    color: #ffffff !important;
}

/* Inputs */
textarea, input {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-radius: 8px;
}

/* Placeholder */
textarea::placeholder {
    color: #666666 !important;
}

/* Botões */
button {
    background-color: #ffffff !important;
    color: #3c3c3c !important;
    border-radius: 10px;
    font-weight: 600;
}
/* Texto interno do botão */
button span, 
button div {
    color: #3c3c3c !important;
}

/* Overlay escuro */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.4);
    z-index: -1;
}

/* Área de upload CSV */
[data-testid="stFileUploader"] {
    background-color: #f5f5f5;
    border-radius: 10px;
}

[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] div {
    color: #3c3c3c !important;
}
[data-testid="stFileUploader"] {
    border: 1px dashed #cccccc;
}
/* Texto padrão do Streamlit (st.write, descrições) */
div[data-testid="stMarkdownContainer"] p {
    color: #ffffff !important;
}
/* Botão Streamlit - texto sempre visível */
.stButton > button {
    background-color: #ffffff !important;
    color: #3c3c3c !important;
    font-weight: 600;
}

.stButton > button span {
    color: #3c3c3c !important;
}


/* Card de resultado */
.result-card {
    background-color: #ffffff !important;
}

.result-card h3 {
    color: #000000 !important;
}

.result-card p,
.result-card strong {
    color: #000000 !important;
}


</style>
""", unsafe_allow_html=True)


# =========================
# INTERFACE
# =========================
st.title("🌱 Análise de Sentimentos")
st.write(
    "Análise de Sentimentos Multilíngue para categorizar as avaliações dos clientes em: "
    "Positivo, Negativo ou Neutro"
)

lang_ui = st.selectbox(
    "Idioma do texto:",
    ["Português - BR", "English - US", "Spanish - 419"]
)

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


# =========================
# MAPEAMENTO DE CLASSES
# =========================
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


# =========================
# AÇÃO
# =========================
use_file = uploaded_file is not None

if st.button("Analisar"):

    if not use_file and text.strip() == "":
        st.warning("Digite um texto ou envie um CSV.")

    else:
        # -------- CSV --------
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
                    label = CLASS_MAPPING[lang].get(
                        raw_label, f"Classe desconhecida: {raw_label}"
                    )
                    results.append((t, label, prob))

                result_df = pd.DataFrame(
                    results, columns=["Texto", "Sentimento", "Confiança"]
                )

                st.dataframe(result_df)

                st.download_button(
                    "Baixar resultados",
                    result_df.to_csv(index=False).encode("utf-8"),
                    file_name="resultado_sentimentos.csv",
                    mime="text/csv"
                )

        # -------- TEXTO ÚNICO --------
        else:
            raw_label, prob = predict(text, lang)
            raw_label = raw_label.strip()
            label = CLASS_MAPPING[lang].get(
                raw_label, f"Classe desconhecida: {raw_label}"
            )

            st.markdown(f"""
            <div class="result-card" style="
                background-color: #ffffff;
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                margin-top: 20px;
            ">
                <h3>Resultado da Análise</h3>
                <p><strong>Sentimento:</strong> {label}</p>
                <p><strong>Confiança:</strong> {prob:.2%}</p>
            </div>
            """, unsafe_allow_html=True)



# =========================
# FOOTER
# =========================
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
© 2026 • Análise de Sentimentos • Todos os Direitos Reservados
</div>
""", unsafe_allow_html=True)
