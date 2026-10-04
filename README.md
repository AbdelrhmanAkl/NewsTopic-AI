# 📰 NewsTopic AI

### NLP · Topic Modeling · Unsupervised Learning

**NewsTopic AI** is an interactive Natural Language Processing application for discovering hidden themes across news articles using two complementary unsupervised learning approaches:

- **Latent Dirichlet Allocation (LDA)**
- **Non-negative Matrix Factorization (NMF)**

The project combines a complete text preprocessing pipeline, trained topic models, interpretable topic labeling, model evaluation, and interactive article analysis through a modern **Streamlit** dashboard.

---

## 🚀 Live Demo

**[Launch NewsTopic AI](https://newstopic-ai.streamlit.app/)**

---

## 📌 Overview

News articles contain large volumes of unstructured text, making it challenging to automatically identify recurring themes and meaningful content patterns.

**NewsTopic AI** addresses this challenge through **topic modeling**, an unsupervised learning technique that discovers latent thematic structures without requiring manually labeled training data.

The application trains and exposes two independent topic modeling approaches:

- **LDA** — probabilistic topic modeling
- **NMF** — matrix-factorization-based topic modeling

Users can explore the discovered topics, inspect their most representative themes, compare both approaches, and analyze new news articles interactively.

---

## ✨ Key Features

- 🧠 LDA-based topic discovery
- 🔬 NMF-based topic discovery
- 📝 Automated English text preprocessing
- 🧹 Stopword and domain-specific news-word filtering
- 🌱 Lemmatization
- 📊 Topic distribution visualization
- 🏷️ Human-readable topic interpretation
- ⚖️ LDA vs NMF model comparison
- 📰 Interactive article analysis
- 🎯 Dominant topic prediction
- 💾 Pre-trained model loading
- ☁️ Streamlit Community Cloud deployment
- 💻 Local inference without model retraining

---

## 🔄 Project Workflow

```text
                    Raw News Articles
                           │
                           ▼
                 ┌───────────────────┐
                 │ Text Preprocessing│
                 └───────────────────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
      Lowercasing      URL Removal   Character Cleaning
            │              │              │
            └──────────────┼──────────────┘
                           ▼
                  Token Filtering
                           │
                           ▼
                      Lemmatization
                           │
                           ▼
                Domain Stopword Removal
                           │
                           ▼
                  Feature Representation
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
                LDA                 NMF
                 │                   │
                 ▼                   ▼
          Topic Discovery     Topic Discovery
                 │                   │
                 └─────────┬─────────┘
                           ▼
                Interactive Dashboard
                           │
                           ▼
                  Article Topic Analysis
```

---

# 🧠 Models

## 1. Latent Dirichlet Allocation — LDA

**Latent Dirichlet Allocation (LDA)** is a probabilistic generative model that represents each document as a mixture of topics and each topic as a probability distribution over words.

The final LDA implementation contains:

- **6 discovered topics**
- Human-readable topic labels
- Topic probability distributions
- **Coherence Score: `0.5987`**

### LDA Topics

| Topic | Interpretation |
|---:|---|
| 1 | Film, Music & Entertainment |
| 2 | Technology, Mobile & Gaming |
| 3 | Politics & Government |
| 4 | Sports & Football |
| 5 | Software, Internet & Digital Technology |
| 6 | Business & Financial Markets |

---

## 2. Non-negative Matrix Factorization — NMF

**Non-negative Matrix Factorization (NMF)** decomposes a non-negative document-term representation into latent components that can be interpreted as topics.

The final NMF implementation contains:

- **10 discovered topics**
- Human-readable topic labels
- Topic weight distributions
- **Reconstruction Error: `44.0819`**

### NMF Topics

| Topic | Interpretation |
|---:|---|
| 1 | Rugby & Six Nations |
| 2 | UK Elections & Political Parties |
| 3 | Economy & Financial Markets |
| 4 | Film & Awards |
| 5 | Mobile, Broadband & Technology |
| 6 | Law, Government & Human Rights |
| 7 | Tennis & International Sports |
| 8 | Russian Oil & Corporate Affairs |
| 9 | Football Clubs & Leagues |
| 10 | Music, Bands & Albums |

---

# ⚖️ Model Comparison

NewsTopic AI provides complementary evaluation signals for both topic modeling approaches.

| Metric | LDA | NMF |
|---|---:|---:|
| Number of Topics | 6 | 10 |
| Evaluation Metric | Coherence | Reconstruction Error |
| Score | `0.5987` | `44.0819` |
| Topic Alignment | `83.01%` | `58.01%` |

> **Note:** LDA coherence and NMF reconstruction error measure different properties of their respective models and should **not** be interpreted as directly equivalent performance metrics.

The comparison is therefore intended to help inspect the different topic structures and behavior of the two approaches rather than declare a universal winner.

---

# 📰 Interactive Article Analysis

One of the main features of NewsTopic AI is the ability to analyze a new article directly through the Streamlit interface.

Users can paste an English news article into the application and receive predictions from both trained topic models.

### Analysis Pipeline

```text
New Article
     │
     ▼
Text Preprocessing
     │
     ▼
Cleaned Representation
     │
     ├───────────────┐
     ▼               ▼
    LDA             NMF
     │               │
     ▼               ▼
Dominant Topic   Dominant Topic
     │               │
     ▼               ▼
Probability       Topic Weights
Distribution      Distribution
     │               │
     └───────┬───────┘
             ▼
       Model Comparison
```

The application:

1. Preprocesses the article.
2. Generates the cleaned text representation.
3. Predicts the dominant LDA topic.
4. Displays the LDA topic probability distribution.
5. Predicts the dominant NMF topic.
6. Displays the NMF topic weights.
7. Allows direct comparison between both approaches.

This transforms the project from a static experimentation notebook into an interactive NLP application suitable for real-time inference.

---

# 📊 Dataset

The project uses a collection of **BBC news articles** for unsupervised topic modeling.

### Dataset Characteristics

- **2,126 news articles**
- English-language news content
- Multiple domains including:
  - Politics
  - Business
  - Technology
  - Sports
  - Entertainment

The raw dataset is intentionally excluded from version control to keep the repository lightweight and avoid unnecessarily committing large data files.

---

# 🧹 Text Preprocessing

A consistent preprocessing pipeline is used during both model development and inference.

### Processing Steps

```text
Raw Text
   │
   ▼
Lowercase Conversion
   │
   ▼
URL Removal
   │
   ▼
Non-Alphabetic Character Removal
   │
   ▼
Tokenization
   │
   ▼
English Stopword Removal
   │
   ▼
Short Token Filtering
   │
   ▼
Lemmatization
   │
   ▼
Domain-Specific News Stopword Removal
   │
   ▼
Processed Text
```

The same preprocessing logic is applied when users submit new articles through the Streamlit application, helping maintain consistency between training and inference.

---

# 🛠️ Technology Stack

## Programming

- Python

## Natural Language Processing

- NLTK
- Gensim
- scikit-learn

## Machine Learning

- Latent Dirichlet Allocation
- Non-negative Matrix Factorization
- TF-IDF / Document-Term Representations

## Data & Visualization

- NumPy
- Pandas
- Plotly

## Application

- Streamlit

## Model Persistence

- Joblib
- Gensim model serialization

## Development

- Jupyter Notebook
- Git
- GitHub

## Deployment

- Streamlit Community Cloud

---

# 📁 Project Structure

```text
NewsTopic-AI/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── README.md
│
├── models/
│   ├── final_lda_model
│   ├── final_lda_model.expElogbeta.npy
│   ├── final_lda_model.id2word
│   ├── final_lda_model.state
│   ├── final_nmf_model.joblib
│   ├── lda_topic_labels.json
│   ├── nmf_topic_labels.json
│   ├── refined_dictionary.pkl
│   └── refined_tfidf_vectorizer.joblib
│
├── notebooks/
│   └── NewsTopic_AI.ipynb
│
└── src/
    ├── preprocessing.py
    ├── lda_model.py
    └── nmf_model.py
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/AbdelrhmanAkl/NewsTopic-AI.git
cd NewsTopic-AI
```

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

## 3. Activate the Environment

### Windows

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## 5. Run the Application

```bash
streamlit run app.py
```

The application will then be available through the local Streamlit URL displayed in the terminal.

---

# 🖥️ Application Sections

## Overview

Provides a high-level summary of the dataset, trained models, discovered topics, and project results.

## LDA Topics

Explores the topics discovered by the LDA model and their associated probability distributions.

## NMF Topics

Explores the topics discovered by the NMF model and their associated topic weights.

## Model Comparison

Provides a side-by-side comparison of the LDA and NMF approaches and their evaluation signals.

## Analyze Article

Allows users to submit a new English news article and inspect its predicted topic from both models.

---

# 🔁 Reproducibility

The project separates **model development** from **application inference**.

The Jupyter Notebook contains the experimentation, preprocessing, and modeling workflow, while the Streamlit application loads the trained artifacts from the `models/` directory.

This architecture allows the deployed application to perform inference without retraining the topic models every time the application starts.

---

# 🚀 Future Improvements

Potential extensions include:

- Automatic topic naming using LLMs
- Dynamic topic-number selection
- Additional topic coherence metrics
- BERTopic comparison
- Transformer-based embeddings
- Multilingual topic modeling
- Larger and more diverse news datasets
- Topic evolution over time
- Document similarity search
- Advanced topic visualization

---

# 💼 Portfolio Highlights

NewsTopic AI demonstrates practical experience in:

- Natural Language Processing
- Unsupervised Machine Learning
- Topic Modeling
- Latent Dirichlet Allocation
- Non-negative Matrix Factorization
- Text Preprocessing
- Model Evaluation
- Model Serialization
- Data Visualization
- Interactive Streamlit Applications
- Machine Learning Inference
- Git & GitHub Workflow
- Cloud Deployment

The project also demonstrates the ability to move from **NLP experimentation → trained models → serialized artifacts → interactive production-style application**.

---

# 👨‍💻 Author

### Abdelrahman Akl

**AI Engineer | AI/ML Instructor | Agentic AI & LLMs**

- **GitHub:** [AbdelrhmanAkl](https://github.com/AbdelrhmanAkl)
- **Live Demo:** [NewsTopic AI](https://newstopic-ai.streamlit.app/)

---

# 📄 License

This project is licensed under the **MIT License**.