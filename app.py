import streamlit as st
import pandas as pd
import base64
from utils import predict


# =====================================================
# CONFIGURAÇÃO DA PÁGINA (sempre no topo)
# =====================================================
st.set_page_config(
    page_title="Análise de Sentimentos",
    page_icon="🌱",
    layout="centered"
)


# =====================================================
# FUNÇÃO PARA FUNDO COM IMAGEM
# =====================================================
def set_background(image_file: str):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        /* Overlay escuro para contraste */
        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.4);
            z-index: -1;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


set_background("fundo.jpg")


# =====================================================
# CSS GLOBAL (CONTROLADO, SEM CONFLITOS)
# =====================================================
st.markdown(
    """
    <style>
    /* ================= TEXTOS DO APP ================= */
    label,
    div[data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }

    /* ================= INPUTS ================= */
    textarea, input {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-radius: 8px;
    }

    textarea::placeholder {
        color: #666666 !important;
    }

    /* ================= BOTÃO ================= */
    .stButton > button {
        background-color: #ffffff !important;
        color: #3c3c3c !important;
        font-weight: 600;
        border-radius: 10px;
    }

    .stButton > button span {
        color: #3c3c3c !important;
    }

    /* ================= UPLOAD CSV ================= */
    [data-testid="stFileUploader"] {
        background-color: #f5f5f5;
        border-radius: 10px;
        border: 1px dashed #cccccc;
    }

    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span {
        color: #3c3c3c !important;
    }

    /* ================= CARD DE RESULTADO ================= */
    .result-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 14px;
        text-align: center;
        margin-top: 24px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }

   /* Força texto do card de resultado (vence MarkdownContainer) */
        div.result-card h3,
        div.result-card p,
        div.result-card strong,
        div.result-card span {
            color: #001969 !important;
}


    /* ================= FOOTER ================= */
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
    """,
    unsafe_allow_html=True
)


# =====================================================
# INTERFACE
# =====================================================
st.title("🌱 Análise de Sentimentos")

st.write(
    "Análise de Sentimentos Multilíngue para categorizar as avaliações dos clientes em: "
    "Positivo, Negativo ou Neutro"
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

st.write("### Ou envie um arquivo CSV")

uploaded_file = st.file_uploader(
    "O CSV deve conter uma coluna chamada 'text'",
    type=["csv"]
)


# =====================================================
# MAPEAMENTO DE CLASSES
# =====================================================
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


# =====================================================
# AÇÃO
# =====================================================
use_file = uploaded_file is not None

if st.button("Analisar"):

    if not use_file and not text.strip():
        st.warning("Digite um texto ou envie um CSV.")
        st.stop()

    # -------- CSV --------
    if use_file:
        df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
        df.columns = df.columns.str.strip().str.lower()

        if "text" not in df.columns:
            st.error("O CSV deve conter uma coluna chamada 'text'.")
            st.stop()

        results = []

        for t in df["text"]:
            raw_label, prob = predict(str(t), lang)
            label = CLASS_MAPPING[lang].get(raw_label.strip(), raw_label)
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
        label = CLASS_MAPPING[lang].get(raw_label.strip(), raw_label)

        st.markdown(
            f"""
            <div class="result-card">
                <h3>Resultado da Análise</h3>
                <p><strong>Sentimento:</strong> {label}</p>
                <p><strong>Confiança:</strong> {prob:.2%}</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# =====================================================
# FOOTER
# =====================================================
st.markdown(
    """
    <div class="footer">
        © 2026 • Análise de Sentimentos • Todos os Direitos Reservados
    </div>
    """,
    unsafe_allow_html=True
)
