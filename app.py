import json
import pickle
import re
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from gensim.models import LdaModel

from src.preprocessing import preprocess_text


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

LDA_COHERENCE = 0.5987
NMF_RECONSTRUCTION_ERROR = 44.0819
LDA_ALIGNMENT = 83.01
NMF_ALIGNMENT = 58.01

TOTAL_ARTICLES = 2126
LDA_TOPIC_COUNT = 6
NMF_TOPIC_COUNT = 10

ACCENT = "#6366f1"        # soft indigo
ACCENT_LIGHT = "#a5b4fc"  # pastel indigo
ACCENT_SOFT = "#eef0ff"   # very light indigo tint

GITHUB_URL = "https://github.com/AbdelrhmanAkl/NewsTopic-AI"
GITHUB_PROFILE = "https://github.com/AbdelrhmanAkl"


st.set_page_config(
    page_title="NewsTopic AI",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)


def html(markup: str):
    """Render HTML safely (strips indentation so Markdown never treats it as code)."""
    st.markdown(
        re.sub(r"^\s+", "", markup, flags=re.M),
        unsafe_allow_html=True,
    )


# ============================================================
# Light & Chic UI
# ============================================================

html(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* ---------- Global ---------- */
    html, body, [class*="css"], .stApp {{
        font-family: 'Inter', -apple-system, 'Segoe UI', Arial, sans-serif;
    }}

    .stApp {{
        background: linear-gradient(180deg, #f7f8ff 0%, #ffffff 340px);
    }}

    .block-container {{
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }}

    /* Hide Streamlit chrome (keeps sidebar toggle working) */
    #MainMenu, footer {{ visibility: hidden; }}
    [data-testid="stToolbar"], [data-testid="stDecoration"],
    [data-testid="stStatusWidget"], .stAppDeployButton {{ display: none !important; }}
    header[data-testid="stHeader"] {{ background: transparent; }}

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {{
        background: #f4f6ff;
        border-right: 1px solid #e6e9fb;
    }}
    section[data-testid="stSidebar"] > div {{ padding-top: 1.6rem; }}

    .sidebar-brand {{
        color: #1e1b4b;
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }}
    .sidebar-brand span {{ color: {ACCENT}; }}
    .sidebar-description {{
        color: #6b7280;
        font-size: 0.82rem;
        line-height: 1.6;
        margin-top: 0.35rem;
    }}
    .sidebar-section {{
        color: #9aa0c3;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-top: 1.6rem;
        margin-bottom: 0.6rem;
    }}
    .sidebar-meta {{
        display: flex;
        justify-content: space-between;
        padding: 0.4rem 0;
        border-bottom: 1px solid #e6e9fb;
        font-size: 0.82rem;
    }}
    .sidebar-meta-label {{ color: #6b7280; }}
    .sidebar-meta-value {{ color: #1e1b4b; font-weight: 700; }}
    .sidebar-link {{
        display: block;
        color: {ACCENT} !important;
        font-size: 0.82rem;
        font-weight: 600;
        text-decoration: none;
        padding: 0.25rem 0;
    }}
    .sidebar-link:hover {{ text-decoration: underline; }}

    /* ---------- Typography ---------- */
    h1, h2, h3, h4 {{ color: #1e1b4b !important; letter-spacing: -0.02em; }}
    p, li, label {{ color: #475569; }}
    .muted {{ color: #6b7280; font-size: 0.9rem; line-height: 1.65; }}

    /* ---------- Hero ---------- */
    .hero {{ padding: 0.5rem 0 1.6rem 0; }}
    .hero-label {{
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        background: {ACCENT_SOFT};
        color: {ACCENT};
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }}
    .hero-title {{
        font-size: 3.1rem;
        font-weight: 800;
        line-height: 1.05;
        letter-spacing: -0.03em;
        color: #1e1b4b;
        margin-bottom: 0.7rem;
    }}
    .hero-title span {{
        background: linear-gradient(90deg, {ACCENT}, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .hero-subtitle {{
        max-width: 760px;
        color: #6b7280;
        font-size: 1.05rem;
        line-height: 1.7;
    }}

    /* ---------- Cards ---------- */
    .stat-card, .topic-card, .info-card, .result-card, .step-card {{
        border: 1px solid #e6e9fb;
        border-radius: 16px;
        background: #ffffff;
        box-shadow: 0 1px 2px rgba(99, 102, 241, 0.04), 0 4px 14px rgba(99, 102, 241, 0.05);
        transition: border-color 0.15s ease, box-shadow 0.15s ease, transform 0.15s ease;
    }}
    .stat-card:hover, .topic-card:hover, .info-card:hover {{
        border-color: {ACCENT_LIGHT};
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.12);
        transform: translateY(-2px);
    }}

    .stat-card {{ padding: 1.15rem 1.25rem; min-height: 112px; }}
    .stat-label {{ color: #6b7280; font-size: 0.8rem; margin-bottom: 0.4rem; }}
    .stat-value {{ color: {ACCENT}; font-size: 1.85rem; font-weight: 800; line-height: 1.1; }}
    .stat-description {{ color: #9ca3af; font-size: 0.75rem; margin-top: 0.4rem; }}

    .topic-card {{ padding: 1.15rem 1.25rem; min-height: 150px; margin-bottom: 1rem; }}
    .topic-number {{ color: {ACCENT}; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; margin-bottom: 0.4rem; }}
    .topic-title {{ color: #1e1b4b; font-size: 1.02rem; font-weight: 700; margin-bottom: 0.7rem; }}
    .topic-words {{ color: #6b7280; font-size: 0.82rem; line-height: 1.8; }}

    .info-card {{ padding: 1.4rem; min-height: 175px; }}
    .info-title {{ color: #1e1b4b; font-size: 1rem; font-weight: 700; margin-bottom: 0.55rem; }}
    .info-text {{ color: #6b7280; font-size: 0.9rem; line-height: 1.7; }}

    /* ---------- Pipeline ---------- */
    .step-card {{ padding: 1rem 1rem; text-align: center; min-height: 112px; }}
    .step-num {{
        display: inline-flex; align-items: center; justify-content: center;
        width: 26px; height: 26px; border-radius: 50%;
        background: {ACCENT_SOFT}; color: {ACCENT};
        font-size: 0.78rem; font-weight: 800; margin-bottom: 0.5rem;
    }}
    .step-title {{ color: #1e1b4b; font-size: 0.9rem; font-weight: 700; }}
    .step-text {{ color: #9ca3af; font-size: 0.76rem; margin-top: 0.25rem; line-height: 1.5; }}

    /* ---------- Results ---------- */
    .result-card {{ padding: 1.3rem 1.4rem; min-height: 190px; }}
    .result-model {{ color: #9aa0c3; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; }}
    .result-badge {{
        display: inline-block; margin: 0.7rem 0 0.9rem 0;
        padding: 0.45rem 0.95rem; border-radius: 999px;
        background: {ACCENT_SOFT}; color: {ACCENT};
        font-size: 1rem; font-weight: 700;
    }}
    .result-metric-label {{ color: #6b7280; font-size: 0.78rem; }}
    .result-metric-value {{ color: #1e1b4b; font-size: 1.5rem; font-weight: 800; }}

    /* ---------- Sections ---------- */
    .section-label {{
        color: {ACCENT}; font-size: 0.74rem; font-weight: 700;
        letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.3rem;
    }}
    .section-title {{ color: #1e1b4b; font-size: 1.6rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0.35rem; }}
    .section-description {{ color: #6b7280; font-size: 0.92rem; margin-bottom: 1.25rem; }}
    .soft-divider {{ height: 1px; background: #e6e9fb; margin: 2.2rem 0; }}

    /* ---------- Inputs ---------- */
    div[data-testid="stTextArea"] textarea {{
        border-radius: 14px; border: 1px solid #dfe3f8; background: #ffffff;
    }}
    div[data-testid="stTextArea"] textarea:focus {{
        border-color: {ACCENT}; box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.12);
    }}
    div[data-baseweb="select"] > div {{ border-radius: 12px; border-color: #dfe3f8; }}

    /* ---------- Buttons ---------- */
    .stButton > button {{
        border-radius: 12px; min-height: 42px; font-weight: 600;
        border: 1px solid #dfe3f8; background: #ffffff; color: #3730a3;
        transition: all 0.15s ease;
    }}
    .stButton > button:hover {{
        border-color: {ACCENT}; color: {ACCENT}; background: {ACCENT_SOFT};
    }}
    .stButton > button[kind="primary"] {{
        background: linear-gradient(90deg, {ACCENT}, #8b5cf6);
        color: #ffffff; border: none;
    }}
    .stButton > button[kind="primary"]:hover {{
        color: #ffffff; filter: brightness(1.06);
        box-shadow: 0 6px 18px rgba(99, 102, 241, 0.3);
    }}

    /* ---------- Metrics ---------- */
    div[data-testid="stMetric"] {{
        background: #ffffff; border: 1px solid #e6e9fb;
        border-radius: 16px; padding: 1rem 1.15rem;
    }}

    /* ---------- Footer ---------- */
    .footer {{ text-align: center; color: #a5a9c4; font-size: 0.78rem; padding-top: 2.5rem; }}
    </style>
    """
)


# ============================================================
# Model Loading
# ============================================================

@st.cache_resource
def load_models():
    """Load trained models and metadata once."""

    lda_model = LdaModel.load(str(MODELS_DIR / "final_lda_model"))
    nmf_model = joblib.load(MODELS_DIR / "final_nmf_model.joblib")
    tfidf_vectorizer = joblib.load(MODELS_DIR / "refined_tfidf_vectorizer.joblib")

    with open(MODELS_DIR / "refined_dictionary.pkl", "rb") as file:
        dictionary = pickle.load(file)

    with open(MODELS_DIR / "lda_topic_labels.json", "r", encoding="utf-8") as file:
        lda_labels = json.load(file)

    with open(MODELS_DIR / "nmf_topic_labels.json", "r", encoding="utf-8") as file:
        nmf_labels = json.load(file)

    return (
        lda_model,
        nmf_model,
        tfidf_vectorizer,
        dictionary,
        lda_labels,
        nmf_labels,
    )


(
    lda_model,
    nmf_model,
    tfidf_vectorizer,
    dictionary,
    lda_labels,
    nmf_labels,
) = load_models()


# ============================================================
# Topic Definitions
# ============================================================

LDA_TOPIC_WORDS = {
    "Film, Music & Entertainment": [
        "film", "best", "show", "world", "award",
        "music", "star", "band", "day", "top",
    ],
    "Technology, Mobile & Gaming": [
        "game", "mobile", "phone", "technology",
        "video", "market", "player", "digital",
        "device", "gaming",
    ],
    "Politics & Government": [
        "government", "labour", "election",
        "minister", "party", "blair", "tory",
        "public", "plan", "police",
    ],
    "Sports & Football": [
        "club", "win", "game", "england",
        "player", "final", "team", "season",
        "champion", "match",
    ],
    "Software, Internet & Digital Technology": [
        "software", "patent", "computer", "online",
        "broadband", "network", "internet", "machine",
        "legal", "law",
    ],
    "Business & Financial Markets": [
        "company", "firm", "market", "month",
        "sale", "share", "price", "business",
        "analyst", "bank",
    ],
}


NMF_TOPIC_WORDS = {
    "Rugby & Six Nations": [
        "england", "wale", "ireland", "rugby",
        "robinson", "six nation", "france",
    ],
    "UK Elections & Political Parties": [
        "labour", "election", "party", "blair",
        "brown", "tory", "tax", "campaign",
    ],
    "Economy & Financial Markets": [
        "growth", "economy", "rate", "price",
        "bank", "market", "profit", "euro",
    ],
    "Film & Awards": [
        "film", "award", "best", "oscar",
        "actor", "festival", "director",
    ],
    "Mobile, Broadband & Technology": [
        "mobile", "phone", "technology", "service",
        "broadband", "network", "digital",
    ],
    "Law, Government & Human Rights": [
        "lord", "law", "government", "police",
        "court", "bill", "human right",
    ],
    "Tennis & International Sports": [
        "final", "open", "champion", "world",
        "seed", "win", "olympic",
    ],
    "Russian Oil & Corporate Affairs": [
        "yukos", "oil", "russian", "gazprom",
        "company", "russia", "court",
    ],
    "Football Clubs & Leagues": [
        "game", "club", "chelsea", "league",
        "liverpool", "united", "arsenal",
    ],
    "Music, Bands & Albums": [
        "music", "band", "album", "chart",
        "song", "rock", "singer",
    ],
}


EXAMPLE_ARTICLES = {
    "Sports": (
        "Chelsea moved clear at the top of the league after a convincing win over "
        "Arsenal at Stamford Bridge. The manager praised his players for their "
        "performance, saying the team had shown great character throughout the "
        "season. Liverpool and Manchester United both dropped points, leaving the "
        "champions with a comfortable lead ahead of the final matches of the season."
    ),
    "Business": (
        "Shares in the company rose sharply after the firm reported a rise in annual "
        "profit and higher sales across its main markets. Analysts said the results "
        "were better than expected, and the bank raised its forecast for growth in "
        "the coming year. The board announced a higher dividend for shareholders "
        "despite rising oil prices and a weaker euro."
    ),
    "Technology": (
        "A new mobile phone with built-in broadband access has been unveiled at a "
        "technology show. The device lets users download music and video over the "
        "network at much faster speeds. Software makers said digital services and "
        "online gaming would drive demand as more people connect to the internet "
        "from their handsets."
    ),
    "Politics": (
        "Labour has launched its election campaign with a pledge on tax and public "
        "services. The prime minister said the government would continue its plans "
        "on policing and health, while the Tory leader attacked the party over "
        "immigration. Party strategists expect the campaign to focus on the economy "
        "in the coming weeks."
    ),
    "Entertainment": (
        "The film won the top prize at the awards ceremony, with its director and "
        "lead actor both collecting trophies. The soundtrack album, which features "
        "a well-known rock band, has also climbed the music chart. Critics called "
        "it the best film of the year, and the festival audience gave it a standing "
        "ovation."
    ),
}


def set_example(text: str):
    st.session_state["article_text"] = text


# ============================================================
# Helper Functions
# ============================================================

def predict_lda(cleaned_text):
    """Generate LDA topic distribution."""

    bow = dictionary.doc2bow(cleaned_text.split())

    distribution = lda_model.get_document_topics(
        bow,
        minimum_probability=0,
    )

    topic_id, probability = max(distribution, key=lambda item: item[1])

    return {
        "topic_id": int(topic_id),
        "label": lda_labels[str(topic_id)],
        "probability": float(probability),
        "distribution": distribution,
    }


def predict_nmf(cleaned_text):
    """Generate NMF topic weights."""

    tfidf_matrix = tfidf_vectorizer.transform([cleaned_text])
    weights = nmf_model.transform(tfidf_matrix)[0]
    topic_id = int(weights.argmax())

    return {
        "topic_id": topic_id,
        "label": nmf_labels[str(topic_id)],
        "weight": float(weights[topic_id]),
        "weights": weights,
    }


def lda_top_words(topic_id, topn=10):
    """Real word weights from the trained LDA model."""
    rows = lda_model.show_topic(topic_id, topn=topn)
    return pd.DataFrame(rows, columns=["Word", "Weight"])


def nmf_top_words(topic_id, topn=10):
    """Real word weights from the trained NMF model."""
    feature_names = tfidf_vectorizer.get_feature_names_out()
    components = nmf_model.components_[topic_id]
    top_idx = components.argsort()[::-1][:topn]
    return pd.DataFrame(
        {
            "Word": [feature_names[i] for i in top_idx],
            "Weight": [float(components[i]) for i in top_idx],
        }
    )


def chart_layout(fig, height=380):
    """Apply a clean, light Plotly layout."""

    fig.update_layout(
        height=height,
        template="simple_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial", color="#475569"),
        margin=dict(l=20, r=30, t=30, b=20),
        xaxis=dict(gridcolor="#eef0fb", zerolinecolor="#eef0fb"),
        yaxis=dict(gridcolor="#eef0fb", zerolinecolor="#eef0fb"),
    )
    return fig


def show_chart(fig):
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
    )


def section_header(label, title, description=None):
    html(f'<div class="section-label">{label}</div>')
    html(f'<div class="section-title">{title}</div>')
    if description:
        html(f'<div class="section-description">{description}</div>')


def divider():
    html('<div class="soft-divider"></div>')


def topic_cards(topic_words):
    columns = st.columns(2)
    for index, (topic_label, words) in enumerate(topic_words.items(), start=1):
        with columns[(index - 1) % 2]:
            word_list = " · ".join(words)
            html(
                f"""
                <div class="topic-card">
                    <div class="topic-number">TOPIC {index:02d}</div>
                    <div class="topic-title">{topic_label}</div>
                    <div class="topic-words">{word_list}</div>
                </div>
                """
            )


def word_weight_chart(df, title):
    df = df.sort_values("Weight", ascending=True)
    fig = px.bar(
        df,
        x="Weight",
        y="Word",
        orientation="h",
        color_discrete_sequence=[ACCENT],
    )
    fig.update_traces(marker_line_width=0, opacity=0.9)
    fig.update_layout(xaxis_title=title, yaxis_title="", showlegend=False)
    return chart_layout(fig, height=400)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    html('<div class="sidebar-brand">News<span>Topic</span> AI</div>')

    html(
        """
        <div class="sidebar-description">
        Unsupervised topic modeling for discovering
        hidden themes in news articles.
        </div>
        """
    )

    html('<div class="sidebar-section">Navigation</div>')

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "LDA Topics",
            "NMF Topics",
            "Model Comparison",
            "Analyze Article",
        ],
        label_visibility="collapsed",
    )

    html('<div class="sidebar-section">Dataset</div>')

    html(
        f"""
        <div class="sidebar-meta">
            <span class="sidebar-meta-label">Articles</span>
            <span class="sidebar-meta-value">{TOTAL_ARTICLES:,}</span>
        </div>
        <div class="sidebar-meta">
            <span class="sidebar-meta-label">LDA topics</span>
            <span class="sidebar-meta-value">{LDA_TOPIC_COUNT}</span>
        </div>
        <div class="sidebar-meta">
            <span class="sidebar-meta-label">NMF topics</span>
            <span class="sidebar-meta-value">{NMF_TOPIC_COUNT}</span>
        </div>
        """
    )

    html('<div class="sidebar-section">Methods</div>')
    st.caption("LDA · Probabilistic Topic Modeling")
    st.caption("NMF · TF-IDF Matrix Factorization")

    html('<div class="sidebar-section">Links</div>')
    html(
        f"""
        <a class="sidebar-link" href="{GITHUB_URL}" target="_blank">↗ Project on GitHub</a>
        <a class="sidebar-link" href="{GITHUB_PROFILE}" target="_blank">↗ Author profile</a>
        """
    )


# ============================================================
# Overview
# ============================================================

if page == "Overview":

    # Hero (Overview only)
    html(
        """
        <div class="hero">
            <div class="hero-label">NLP · Topic Modeling</div>
            <div class="hero-title">News<span>Topic</span> AI</div>
            <div class="hero-subtitle">
                Discover hidden themes in news articles using
                two complementary unsupervised learning approaches:
                Latent Dirichlet Allocation and Non-negative Matrix
                Factorization.
            </div>
        </div>
        """
    )

    section_header(
        "Overview",
        "Project at a glance",
        "A compact NLP system for automatically discovering recurring "
        "semantic structures in a collection of news articles.",
    )

    stats = [
        ("Articles analyzed", f"{TOTAL_ARTICLES:,}", "News documents"),
        ("LDA topics", f"{LDA_TOPIC_COUNT}", "Compact topic structure"),
        ("NMF topics", f"{NMF_TOPIC_COUNT}", "Fine-grained structure"),
        ("LDA coherence", f"{LDA_COHERENCE:.4f}", "Topic quality metric"),
    ]

    for col, (label, value, desc) in zip(st.columns(4), stats):
        with col:
            html(
                f"""
                <div class="stat-card">
                    <div class="stat-label">{label}</div>
                    <div class="stat-value">{value}</div>
                    <div class="stat-description">{desc}</div>
                </div>
                """
            )

    divider()

    # Pipeline
    section_header(
        "Pipeline",
        "From raw text to topics",
        "Every article goes through the same five steps.",
    )

    steps = [
        ("Raw text", "News article input"),
        ("Preprocessing", "Cleaning, tokenizing, lemmatizing"),
        ("Vectorization", "Bag-of-words and TF-IDF"),
        ("Modeling", "LDA and NMF"),
        ("Topics", "Dominant theme and weights"),
    ]

    for col, (index, (title, text)) in zip(
        st.columns(5), enumerate(steps, start=1)
    ):
        with col:
            html(
                f"""
                <div class="step-card">
                    <div class="step-num">{index}</div>
                    <div class="step-title">{title}</div>
                    <div class="step-text">{text}</div>
                </div>
                """
            )

    divider()

    # How it works
    section_header("Approach", "How it works")

    col1, col2 = st.columns(2)

    with col1:
        html(
            """
            <div class="info-card">
                <div class="info-title">LDA · Probabilistic Modeling</div>
                <div class="info-text">
                    LDA represents each document as a mixture of
                    latent topics and each topic as a distribution
                    over words. The refined model produces a compact
                    six-topic representation of the dataset.
                </div>
            </div>
            """
        )

    with col2:
        html(
            """
            <div class="info-card">
                <div class="info-title">NMF · Matrix Factorization</div>
                <div class="info-text">
                    NMF decomposes the TF-IDF document-term matrix
                    into interpretable topic components. Its
                    ten-topic structure captures more granular
                    subtopics across sports, technology, politics,
                    entertainment and business.
                </div>
            </div>
            """
        )

    divider()

    # Model summary
    section_header("Evaluation", "Model summary")

    summary_df = pd.DataFrame(
        {
            "Model": ["LDA", "NMF"],
            "Topics": [LDA_TOPIC_COUNT, NMF_TOPIC_COUNT],
            "Primary Metric": ["Coherence", "Reconstruction Error"],
            "Score": [LDA_COHERENCE, NMF_RECONSTRUCTION_ERROR],
            "Category Alignment": [LDA_ALIGNMENT, NMF_ALIGNMENT],
        }
    )

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.NumberColumn(format="%.4f"),
            "Category Alignment": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )


# ============================================================
# LDA Topics
# ============================================================

elif page == "LDA Topics":

    section_header(
        "Topic Explorer",
        "LDA Topics",
        "Six broad semantic themes discovered by the refined LDA model.",
    )

    topic_cards(LDA_TOPIC_WORDS)

    divider()

    section_header(
        "Word weights",
        "Inside a topic",
        "Top words and their weights, taken directly from the trained model.",
    )

    lda_ids = sorted(int(key) for key in lda_labels.keys())

    selected = st.selectbox(
        "Choose a topic",
        lda_ids,
        format_func=lambda i: f"Topic {i + 1:02d} · {lda_labels[str(i)]}",
    )

    show_chart(
        word_weight_chart(lda_top_words(selected), "Word probability")
    )


# ============================================================
# NMF Topics
# ============================================================

elif page == "NMF Topics":

    section_header(
        "Topic Explorer",
        "NMF Topics",
        "Ten fine-grained themes discovered from the TF-IDF representation using NMF.",
    )

    topic_cards(NMF_TOPIC_WORDS)

    divider()

    section_header(
        "Word weights",
        "Inside a topic",
        "Top terms and their component weights, taken directly from the trained model.",
    )

    nmf_ids = sorted(int(key) for key in nmf_labels.keys())

    selected = st.selectbox(
        "Choose a topic",
        nmf_ids,
        format_func=lambda i: f"Topic {i + 1:02d} · {nmf_labels[str(i)]}",
    )

    show_chart(
        word_weight_chart(nmf_top_words(selected), "Component weight")
    )


# ============================================================
# Model Comparison
# ============================================================

elif page == "Model Comparison":

    section_header(
        "Evaluation",
        "LDA vs NMF",
        "Comparing topic granularity and alignment with the original BBC news categories.",
    )

    comparison_df = pd.DataFrame(
        {
            "Model": ["LDA", "NMF"],
            "Category Alignment": [LDA_ALIGNMENT, NMF_ALIGNMENT],
        }
    )

    fig = px.bar(
        comparison_df,
        x="Model",
        y="Category Alignment",
        text="Category Alignment",
        color="Model",
        color_discrete_map={"LDA": ACCENT, "NMF": ACCENT_LIGHT},
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        marker_line_width=0,
    )

    fig.update_layout(
        xaxis_title="",
        yaxis_title="Category Alignment (%)",
        yaxis_range=[0, 100],
        showlegend=False,
        bargap=0.55,
    )

    show_chart(chart_layout(fig, height=420))

    col1, col2 = st.columns(2)

    with col1:
        st.metric("LDA Coherence", f"{LDA_COHERENCE:.4f}")
        html(
            """
            <div class="muted">
            LDA provides a compact topic structure with strong
            semantic coherence and broad category coverage.
            </div>
            """
        )

    with col2:
        st.metric("NMF Reconstruction Error", f"{NMF_RECONSTRUCTION_ERROR:.4f}")
        html(
            """
            <div class="muted">
            NMF provides a more granular representation, separating
            closely related areas such as football, rugby, tennis,
            film and music.
            </div>
            """
        )

    divider()

    html('<div class="section-label">Interpretation</div>')

    st.markdown(
        """
        **LDA** is better suited for a compact representation of
        broad semantic categories, while **NMF** exposes more
        specific subtopics through the TF-IDF representation.
        The two approaches are therefore complementary rather than
        strictly interchangeable.
        """
    )


# ============================================================
# Analyze Article
# ============================================================

elif page == "Analyze Article":

    section_header(
        "Interactive NLP",
        "Analyze an article",
        "Paste an English news article, or try one of the examples below, "
        "and compare the dominant topic predicted by both models.",
    )

    st.caption("Quick examples")

    example_columns = st.columns(len(EXAMPLE_ARTICLES))

    for column, (name, text) in zip(example_columns, EXAMPLE_ARTICLES.items()):
        with column:
            st.button(
                name,
                key=f"example_{name}",
                use_container_width=True,
                on_click=set_example,
                args=(text,),
            )

    article = st.text_area(
        "Article text",
        key="article_text",
        height=240,
        placeholder="Paste an English news article here...",
    )

    analyze = st.button(
        "Run Topic Analysis",
        type="primary",
        use_container_width=True,
    )

    if analyze:

        if not article.strip():

            st.warning("Please enter an article before running the analysis.")

        else:

            cleaned = preprocess_text(article)

            if not cleaned.strip():

                st.error(
                    "The article does not contain enough usable "
                    "English text after preprocessing."
                )

            else:

                lda_result = predict_lda(cleaned)
                nmf_result = predict_nmf(cleaned)

                divider()

                section_header("Results", "Topic predictions")

                col1, col2 = st.columns(2)

                with col1:
                    html(
                        f"""
                        <div class="result-card">
                            <div class="result-model">LDA · Probabilistic</div>
                            <div class="result-badge">{lda_result["label"]}</div>
                            <div class="result-metric-label">Topic probability</div>
                            <div class="result-metric-value">{lda_result["probability"]:.2%}</div>
                        </div>
                        """
                    )

                with col2:
                    html(
                        f"""
                        <div class="result-card">
                            <div class="result-model">NMF · Matrix Factorization</div>
                            <div class="result-badge">{nmf_result["label"]}</div>
                            <div class="result-metric-label">Topic weight</div>
                            <div class="result-metric-value">{nmf_result["weight"]:.4f}</div>
                        </div>
                        """
                    )

                divider()

                # LDA distribution
                html('<div class="section-title">LDA topic distribution</div>')

                lda_distribution = pd.DataFrame(
                    [
                        {
                            "Topic": lda_labels[str(topic_id)],
                            "Probability": probability,
                        }
                        for topic_id, probability in lda_result["distribution"]
                    ]
                ).sort_values("Probability", ascending=True)

                fig = px.bar(
                    lda_distribution,
                    x="Probability",
                    y="Topic",
                    orientation="h",
                    text="Probability",
                    color_discrete_sequence=[ACCENT],
                )

                fig.update_traces(
                    texttemplate="%{text:.1%}",
                    textposition="outside",
                    marker_line_width=0,
                    opacity=0.9,
                )

                fig.update_layout(
                    xaxis_title="Probability",
                    yaxis_title="",
                    xaxis_tickformat=".0%",
                    showlegend=False,
                )

                show_chart(chart_layout(fig, height=420))

                # NMF distribution
                html('<div class="section-title">NMF topic weights</div>')

                nmf_topic_names = [
                    nmf_labels[str(i)] for i in range(len(nmf_result["weights"]))
                ]

                nmf_distribution = pd.DataFrame(
                    {
                        "Topic": nmf_topic_names,
                        "Weight": nmf_result["weights"],
                    }
                ).sort_values("Weight", ascending=True)

                fig = px.bar(
                    nmf_distribution,
                    x="Weight",
                    y="Topic",
                    orientation="h",
                    text="Weight",
                    color_discrete_sequence=[ACCENT_LIGHT],
                )

                fig.update_traces(
                    texttemplate="%{text:.4f}",
                    textposition="outside",
                    marker_line_width=0,
                )

                fig.update_layout(
                    xaxis_title="Topic Weight",
                    yaxis_title="",
                    showlegend=False,
                )

                show_chart(chart_layout(fig, height=520))

                with st.expander("View processed text"):
                    st.code(cleaned, language="text")


# ============================================================
# Footer
# ============================================================

html(
    """
    <div class="footer">
        NewsTopic AI · NLP Topic Modeling · LDA + NMF
    </div>
    """
)
