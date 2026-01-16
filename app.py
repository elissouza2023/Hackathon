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
# CSS PERSONALIZADO (corrigido e mais controlado)
# =====================================================
st.markdown(
    """
    <style>
    /* Fundo escuro semitransparente nos containers principais */
    .main > div {
        background-color: rgba(0, 0, 0, 0.4);
        padding: 2rem;
        border-radius: 15px;
        backdrop-filter: blur(5px);
    }

    /* Títulos e textos brancos */
    h1, h2, h3, .stMarkdown, label, .stSelectbox > div > div, .stTextArea label {
        color: white !important;
    }

    /* Botão Analisar visível e estilizado */
    div.stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 8px;
        height: auto;
        width: 100%;
        font-size: 1.2rem;
    }
    div.stButton > button:hover {
        background-color: #45a049;
    }

    /* Ajuste do tamanho do subtítulo "Ou envie um arquivo CSV" */
    .csv-subtitle h3 {
        font-size: 1.1rem !important;
        margin-top: 2rem;
        margin-bottom: 0.8rem;
    }

    /* File uploader mais visível */
    .uploadedFile {
        color: white;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.9rem;
        margin-top: 4rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# INTERFACE PRINCIPAL
# =====================================================
st.title("🌱 Análise de Sentimentos")

st.markdown(
    "Análise de Sentimentos Multilíngue para categorizar as avaliações dos clientes em: "
    "**Positivo**, **Negativo** ou **Neutro**"
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

# Subtítulo com classe personalizada para controlar o tamanho
st.markdown('<div class="csv-subtitle"><h3>Ou envie um arquivo CSV</h3></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "O CSV deve conter uma coluna chamada 'text'",
    type=["csv"],
    help="Máximo 200MB • Apenas arquivos .csv"
)

# =====================================================
# MAPEAMENTO DE CLASSES
# =====================================================
CLASS_MAPPING = {
    "pt": {"Positivo": "Positivo", "Negativo": "Negativo", "Neutro": "Neutro"},
    "es": {"Positivo": "Positivo", "Negativo": "Negativo", "Neutral": "Neutro"},
    "en": {"Positive": "Positivo", "Negative": "Negativo", "Neutral": "Neutro"}
}

# =====================================================
# BOTÃO ANALISAR (agora bem visível)
# =====================================================
use_file = uploaded_file is not None

if st.button("Analisar"):
    if not use_file and not text.strip():
        st.warning("⚠️ Digite um texto ou envie um arquivo CSV para analisar.")
        st.stop()

    # Processamento do CSV
    if use_file:
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
            df.columns = df.columns.str.strip().str.lower()
            if "text" not in df.columns:
                st.error("❌ O CSV deve conter uma coluna chamada exatamente 'text'.")
                st.stop()

            results = []
            for t in df["text"]:
                raw_label, prob = predict(str(t), lang)
                label = CLASS_MAPPING[lang].get(raw_label.strip(), raw_label)
                results.append((t, label, f"{prob:.2%}"))

            result_df = pd.DataFrame(results, columns=["Texto", "Sentimento", "Confiança"])
            st.success("✅ Análise concluída!")
            st.dataframe(result_df, use_container_width=True)

            csv_data = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Baixar resultados como CSV",
                data=csv_data,
                file_name="resultado_sentimentos.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Erro ao processar o CSV: {e}")

    # Processamento de texto único
    else:
        raw_label, prob = predict(text, lang)
        label = CLASS_MAPPING[lang].get(raw_label.strip(), raw_label)
        
        # Resultado destacado
        st.markdown(
            f"""
            <div style="background-color: rgba(0,0,0,0.5); padding: 2rem; border-radius: 15px; text-align: center; margin-top: 2rem;">
                <h2 style="color: white; margin-bottom: 1rem;">Resultado da Análise</h2>
                <h3 style="color: {'#4CAF50' if label == 'Positivo' else '#f44336' if label == 'Negativo' else '#ff9800'};">
                    Sentimento: {label}
                </h3>
                <h4 style="color: white;">
                    Confiança: {prob:.2%}
                </h4>
            </div>
            """,
            unsafe_allow_html=True
        )

# =====================================================
# FOOTER
# =====================================================
st.markdown('<div class="footer">© 2026 • Análise de Sentimentos • Todos os Direitos Reservados</div>', unsafe_allow_html=True)
