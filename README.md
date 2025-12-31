# Análise de Sentimentos Multilíngue (PT / ES / EN)

Este projeto contém notebooks independentes para **análise de sentimentos** em três idiomas:

- Português
- Espanhol
- Inglês

Cada notebook implementa um pipeline completo de NLP clássico, desde o pré-processamento até a avaliação dos modelos, respeitando as particularidades linguísticas de cada idioma.

---

## 📁 Estrutura do projeto
```
├── notebooks/
│   ├── notebook.ipynb    # Português
│   ├── notebook-es.ipynb    # Espanhol
│   └── notebook-en.ipynb    # Inglês
├── models/
│   ├── modelo_lr_sentimentos_es.pkl
│   ├── modelo_svm_sentimentos_es.pkl
│   └── tfidf_es.pkl
├── modelo_lr_sentimentos_en.pkl
│   ├── modelo_svm_sentimentos_en.pkl
│   └── tfidf_en.pkl
├── modelo_lr_sentimentos.pkl
│   ├── modelo_svm_sentimentos.pkl
│   └── tfidf.pkl
├── data/
│   └── data.csv
│   └── data-es.csv
│   └── datas-en.csv
├── requirements.txt
└── README.md
```

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
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Abra o notebook correspondente ao idioma desejado:
   - `notebook.ipynb` (Português)
   - `notebook-es.ipynb` (Espanhol)
   - `notebook-en.ipynb` (Inglês)

4. Execute as células em ordem.

---

## 📌 Observações importantes

- Os notebooks foram projetados para datasets monolíngues.
- Não é utilizada tradução automática.
- O foco do projeto é:
  - Interpretabilidade
  - Simplicidade
  - Baixo custo computacional

---

## 📄 Licença
Projeto de uso acadêmico e experimental.
