# Análise de Sentimentos Multilíngue (PT / ES / EN)

Este projeto contém notebooks independentes para **análise de sentimentos** em três idiomas:

- Português
- Espanhol
- Inglês

Cada notebook implementa um pipeline completo de NLP clássico, desde o pré-processamento até a avaliação dos modelos, respeitando as particularidades linguísticas de cada idioma.

---

## 📁 Estrutura do projeto
├── notebooks/
│   ├── sentiment_pt.ipynb        # Português
│   ├── sentiment_es.ipynb        # Espanhol
│   └── sentiment_en.ipynb        # Inglês
├── models/
│   ├── modelo_lr_sentimentos_es.pkl
│   ├── modelo_svm_sentimentos_es.pkl
│   └── tfidf_es.pkl
├── data/
│   └── dataset_es.csv
├── requirements.txt
└── README.md



## 🧠 Metodologia

Em cada idioma é utilizado o mesmo enfoque geral:

- Limpeza e normalização de texto
- Tratamento explícito de:
  - Negações
  - Intensificadores
  - Eventos negativos
- Vetorização com TF-IDF
- Modelos de classificação:
  - Regressão Logística
  - SVM linear
- Avaliação com:
  - Acurácia
  - Classification Report
  - Matriz de confusão

As regras de pré-processamento são adaptadas especificamente para cada idioma.

## 📊 Modelos incluídos

- Regressão Logística
- Support Vector Machine (SVM linear)

Os modelos treinados e o vetorizador TF-IDF são armazenados na pasta `models/` para reutilização e possíveis etapas de deploy.

## ▶️ Como usar

1. Clone o repositório:

   ```bash
   git clone https://github.com/seu-usuario/seu-repositorio.git

2. Instale as dependências:

  ```bash
    pip install -r requirements.txt

3. Abra o notebook correspondente ao idioma desejado:

  ```bash
sentiment_pt.ipynb (Português)
sentiment_es.ipynb (Espanhol)
sentiment_en.ipynb (Inglês)

4. Execute as células em ordem.

---

## 📌 Observações importantes

Os notebooks foram projetados para datasets monolíngues.
Não é utilizada tradução automática.
O foco do projeto é:
Interpretabilidade
Simplicidade
Baixo custo computacional

---


## 🚀 Próximos passos (opcional)

Integração com Streamlit
Visualizações de interpretabilidade (SHAP)
Unificação do pipeline em módulos reutilizáveis
Comparação de desempenho entre idiomas

---

## 📄 Licença
Projeto de uso acadêmico e experimental.
