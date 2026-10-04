import json
import pickle
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


st.set_page_config(
    page_title="NewsTopic AI",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Premium Minimal UI
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- Global ---------- */

    .stApp {
        background: #ffffff;
    }

    .block-container {
        max-width: 1280px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }

    section[data-testid="stSidebar"] {
        background: #fafafa;
        border-right: 1px solid #eeeeee;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }

    /* ---------- Typography ---------- */

    h1, h2, h3 {
        color: #171717 !important;
        letter-spacing: -0.025em;
    }

    p, li, span, label {
        color: #4b4b4b;
    }

    .muted {
        color: #737373;
        font-size: 0.92rem;
        line-height: 1.6;
    }

    /* ---------- Header ---------- */

    .hero {
        padding: 0.5rem 0 2rem 0;
    }

    .hero-label {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border: 1px solid #e5e5e5;
        border-radius: 999px;
        background: #fafafa;
        color: #525252;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 750;
        line-height: 1.05;
        color: #171717;
        margin-bottom: 0.7rem;
    }

    .hero-subtitle {
        max-width: 760px;
        color: #737373;
        font-size: 1.05rem;
        line-height: 1.7;
    }

    /* ---------- Cards ---------- */

    .stat-card {
        border: 1px solid #e8e8e8;
        border-radius: 14px;
        padding: 1.15rem 1.25rem;
        background: #ffffff;
        min-height: 105px;
    }

    .stat-label {
        color: #737373;
        font-size: 0.82rem;
        margin-bottom: 0.45rem;
    }

    .stat-value {
        color: #171717;
        font-size: 1.7rem;
        font-weight: 700;
        line-height: 1.1;
    }

    .stat-description {
        color: #8a8a8a;
        font-size: 0.76rem;
        margin-top: 0.4rem;
    }

    .topic-card {
        border: 1px solid #e8e8e8;
        border-radius: 14px;
        padding: 1.15rem 1.2rem;
        background: #ffffff;
        min-height: 150px;
        transition: border-color 0.15s ease;
    }

    .topic-card:hover {
        border-color: #cfcfcf;
    }

    .topic-number {
        color: #999999;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }

    .topic-title {
        color: #202020;
        font-size: 1.02rem;
        font-weight: 650;
        margin-bottom: 0.7rem;
    }

    .topic-words {
        color: #737373;
        font-size: 0.82rem;
        line-height: 1.7;
    }

    .info-card {
        border: 1px solid #e8e8e8;
        border-radius: 14px;
        padding: 1.4rem;
        background: #ffffff;
        height: 100%;
    }

    .info-title {
        color: #202020;
        font-size: 1rem;
        font-weight: 650;
        margin-bottom: 0.55rem;
    }

    .info-text {
        color: #737373;
        font-size: 0.9rem;
        line-height: 1.7;
    }

    /* ---------- Section ---------- */

    .section-label {
        color: #8a8a8a;
        font-size: 0.75rem;
        font-weight: 650;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }

    .section-title {
        color: #171717;
        font-size: 1.55rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    .section-description {
        color: #737373;
        font-size: 0.9rem;
        margin-bottom: 1.25rem;
    }

    /* ---------- Divider ---------- */

    .soft-divider {
        height: 1px;
        background: #eeeeee;
        margin: 2rem 0;
    }

    /* ---------- Sidebar ---------- */

    .sidebar-brand {
        color: #171717;
        font-size: 1.35rem;
        font-weight: 750;
        letter-spacing: -0.02em;
    }

    .sidebar-description {
        color: #858585;
        font-size: 0.82rem;
        line-height: 1.6;
        margin-top: 0.35rem;
    }

    .sidebar-section {
        color: #999999;
        font-size: 0.72rem;
        font-weight: 650;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-top: 1.6rem;
        margin-bottom: 0.65rem;
    }

    .sidebar-meta {
        display: flex;
        justify-content: space-between;
        padding: 0.35rem 0;
        border-bottom: 1px solid #eeeeee;
        font-size: 0.8rem;
    }

    .sidebar-meta:last-child {
        border-bottom: none;
    }

    .sidebar-meta-label {
        color: #858585;
    }

    .sidebar-meta-value {
        color: #333333;
        font-weight: 600;
    }

    /* ---------- Inputs ---------- */

    div[data-testid="stTextArea"] textarea {
        border-radius: 12px;
        border: 1px solid #dddddd;
        background: #ffffff;
    }

    div[data-testid="stTextArea"] textarea:focus {
        border-color: #a3a3a3;
        box-shadow: none;
    }

    /* ---------- Buttons ---------- */

    .stButton > button {
        border-radius: 10px;
        min-height: 42px;
        font-weight: 600;
        border: 1px solid #d9d9d9;
    }

    /* ---------- Metrics ---------- */

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e8e8e8;
        border-radius: 14px;
        padding: 1rem 1.15rem;
    }

    div[data-testid="stMetricLabel"] {
        color: #737373;
    }

    div[data-testid="stMetricValue"] {
        color: #171717;
    }

    /* ---------- Footer ---------- */

    .footer {
        text-align: center;
        color: #a0a0a0;
        font-size: 0.78rem;
        padding-top: 2.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Model Loading
# ============================================================

@st.cache_resource
def load_models():
    """Load trained models and metadata once."""

    lda_model = LdaModel.load(
        str(MODELS_DIR / "final_lda_model")
    )

    nmf_model = joblib.load(
        MODELS_DIR / "final_nmf_model.joblib"
    )

    tfidf_vectorizer = joblib.load(
        MODELS_DIR / "refined_tfidf_vectorizer.joblib"
    )

    with open(
        MODELS_DIR / "refined_dictionary.pkl",
        "rb",
    ) as file:
        dictionary = pickle.load(file)

    with open(
        MODELS_DIR / "lda_topic_labels.json",
        "r",
        encoding="utf-8",
    ) as file:
        lda_labels = json.load(file)

    with open(
        MODELS_DIR / "nmf_topic_labels.json",
        "r",
        encoding="utf-8",
    ) as file:
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

    topic_id, probability = max(
        distribution,
        key=lambda item: item[1],
    )

    return {
        "topic_id": int(topic_id),
        "label": lda_labels[str(topic_id)],
        "probability": float(probability),
        "distribution": distribution,
    }


def predict_nmf(cleaned_text):
    """Generate NMF topic weights."""

    tfidf_matrix = tfidf_vectorizer.transform(
        [cleaned_text]
    )

    weights = nmf_model.transform(
        tfidf_matrix
    )[0]

    topic_id = int(weights.argmax())

    return {
        "topic_id": topic_id,
        "label": nmf_labels[str(topic_id)],
        "weight": float(weights[topic_id]),
        "weights": weights,
    }


def chart_layout(fig, height=380):
    """Apply a clean portfolio-friendly Plotly layout."""

    fig.update_layout(
        height=height,
        template="simple_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(
            family="Arial",
            color="#444444",
        ),
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=20,
        ),
        title=dict(
            font=dict(
                size=17,
                color="#202020",
            ),
        ),
        xaxis=dict(
            gridcolor="#eeeeee",
            zerolinecolor="#eeeeee",
        ),
        yaxis=dict(
            gridcolor="#eeeeee",
            zerolinecolor="#eeeeee",
        ),
    )

    return fig


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-brand">NewsTopic AI</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-description">
        Unsupervised topic modeling for discovering
        hidden themes in news articles.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-section">Navigation</div>',
        unsafe_allow_html=True,
    )

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

    st.markdown(
        '<div class="sidebar-section">Dataset</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
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
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-section">Methods</div>',
        unsafe_allow_html=True,
    )

    st.caption("LDA · Probabilistic Topic Modeling")
    st.caption("NMF · TF-IDF Matrix Factorization")


# ============================================================
# Hero Header
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-label">
            NLP · Topic Modeling
        </div>
        <div class="hero-title">
            NewsTopic AI
        </div>
        <div class="hero-subtitle">
            Discover hidden themes in news articles using
            two complementary unsupervised learning approaches:
            Latent Dirichlet Allocation and Non-negative Matrix
            Factorization.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Overview
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="section-label">Overview</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Project at a glance</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
        A compact NLP system for automatically discovering
        recurring semantic structures in a collection of news articles.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Articles analyzed</div>
                <div class="stat-value">{TOTAL_ARTICLES:,}</div>
                <div class="stat-description">News documents</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">LDA topics</div>
                <div class="stat-value">{LDA_TOPIC_COUNT}</div>
                <div class="stat-description">Compact topic structure</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">NMF topics</div>
                <div class="stat-value">{NMF_TOPIC_COUNT}</div>
                <div class="stat-description">Fine-grained structure</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">LDA coherence</div>
                <div class="stat-value">{LDA_COHERENCE:.4f}</div>
                <div class="stat-description">Topic quality metric</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="soft-divider"></div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Project Explanation
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-label">Approach</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">How it works</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-title">
                    LDA · Probabilistic Modeling
                </div>
                <div class="info-text">
                    LDA represents each document as a mixture of
                    latent topics and each topic as a distribution
                    over words. The refined model produces a compact
                    six-topic representation of the dataset.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-title">
                    NMF · Matrix Factorization
                </div>
                <div class="info-text">
                    NMF decomposes the TF-IDF document-term matrix
                    into interpretable topic components. Its
                    ten-topic structure captures more granular
                    subtopics across sports, technology, politics,
                    entertainment and business.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="soft-divider"></div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Model Summary
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-label">Evaluation</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Model summary</div>',
        unsafe_allow_html=True,
    )

    comparison_df = pd.DataFrame(
        {
            "Model": ["LDA", "NMF"],
            "Topics": [LDA_TOPIC_COUNT, NMF_TOPIC_COUNT],
            "Primary Metric": [
                "Coherence",
                "Reconstruction Error",
            ],
            "Score": [
                LDA_COHERENCE,
                NMF_RECONSTRUCTION_ERROR,
            ],
            "Category Alignment": [
                LDA_ALIGNMENT,
                NMF_ALIGNMENT,
            ],
        }
    )

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.NumberColumn(
                format="%.4f"
            ),
            "Category Alignment": st.column_config.NumberColumn(
                format="%.2f%%"
            ),
        },
    )


# ============================================================
# LDA Topics
# ============================================================

elif page == "LDA Topics":

    st.markdown(
        '<div class="section-label">Topic Explorer</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">LDA Topics</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
        Six broad semantic themes discovered by the refined
        LDA model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    columns = st.columns(2)

    for index, (topic_label, words) in enumerate(
        LDA_TOPIC_WORDS.items(),
        start=1,
    ):

        with columns[(index - 1) % 2]:

            word_list = " · ".join(words)

            st.markdown(
                f"""
                <div class="topic-card">
                    <div class="topic-number">
                        TOPIC {index:02d}
                    </div>
                    <div class="topic-title">
                        {topic_label}
                    </div>
                    <div class="topic-words">
                        {word_list}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# NMF Topics
# ============================================================

elif page == "NMF Topics":

    st.markdown(
        '<div class="section-label">Topic Explorer</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">NMF Topics</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
        Ten fine-grained themes discovered from the TF-IDF
        representation using NMF.
        </div>
        """,
        unsafe_allow_html=True,
    )

    columns = st.columns(2)

    for index, (topic_label, words) in enumerate(
        NMF_TOPIC_WORDS.items(),
        start=1,
    ):

        with columns[(index - 1) % 2]:

            word_list = " · ".join(words)

            st.markdown(
                f"""
                <div class="topic-card">
                    <div class="topic-number">
                        TOPIC {index:02d}
                    </div>
                    <div class="topic-title">
                        {topic_label}
                    </div>
                    <div class="topic-words">
                        {word_list}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# Model Comparison
# ============================================================

elif page == "Model Comparison":

    st.markdown(
        '<div class="section-label">Evaluation</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">LDA vs NMF</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
        Comparing topic granularity and alignment with the
        original BBC news categories.
        </div>
        """,
        unsafe_allow_html=True,
    )

    comparison_df = pd.DataFrame(
        {
            "Model": ["LDA", "NMF"],
            "Category Alignment": [
                LDA_ALIGNMENT,
                NMF_ALIGNMENT,
            ],
        }
    )

    fig = px.bar(
        comparison_df,
        x="Model",
        y="Category Alignment",
        text="Category Alignment",
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
    )

    fig.update_layout(
        xaxis_title="",
        yaxis_title="Category Alignment (%)",
        yaxis_range=[0, 100],
        showlegend=False,
    )

    fig = chart_layout(fig, height=420)

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
        },
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "LDA Coherence",
            f"{LDA_COHERENCE:.4f}",
        )

        st.markdown(
            """
            <div class="muted">
            LDA provides a compact topic structure with strong
            semantic coherence and broad category coverage.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.metric(
            "NMF Reconstruction Error",
            f"{NMF_RECONSTRUCTION_ERROR:.4f}",
        )

        st.markdown(
            """
            <div class="muted">
            NMF provides a more granular representation, separating
            closely related areas such as football, rugby, tennis,
            film and music.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="soft-divider"></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-label">Interpretation</div>',
        unsafe_allow_html=True,
    )

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

    st.markdown(
        '<div class="section-label">Interactive NLP</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Analyze an article</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
        Paste an English news article and compare its dominant
        topic predictions from both trained models.
        </div>
        """,
        unsafe_allow_html=True,
    )

    article = st.text_area(
        "Article text",
        height=250,
        placeholder=(
            "Paste an English news article here..."
        ),
        label_visibility="visible",
    )

    analyze = st.button(
        "Run Topic Analysis",
        type="primary",
        use_container_width=True,
    )

    if analyze:

        if not article.strip():

            st.warning(
                "Please enter an article before running the analysis."
            )

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

                st.markdown(
                    '<div class="soft-divider"></div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    '<div class="section-label">Results</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    '<div class="section-title">Topic predictions</div>',
                    unsafe_allow_html=True,
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.markdown(
                        "#### LDA"
                    )

                    st.metric(
                        "Dominant Topic",
                        lda_result["label"],
                    )

                    st.metric(
                        "Topic Probability",
                        f"{lda_result['probability']:.2%}",
                    )

                with col2:

                    st.markdown(
                        "#### NMF"
                    )

                    st.metric(
                        "Dominant Topic",
                        nmf_result["label"],
                    )

                    st.metric(
                        "Topic Weight",
                        f"{nmf_result['weight']:.4f}",
                    )

                st.markdown(
                    '<div class="soft-divider"></div>',
                    unsafe_allow_html=True,
                )

                # ------------------------------------------------
                # LDA Distribution
                # ------------------------------------------------

                st.markdown(
                    '<div class="section-title">LDA topic distribution</div>',
                    unsafe_allow_html=True,
                )

                lda_distribution = pd.DataFrame(
                    [
                        {
                            "Topic": lda_labels[str(topic_id)],
                            "Probability": probability,
                        }
                        for topic_id, probability
                        in lda_result["distribution"]
                    ]
                ).sort_values(
                    "Probability",
                    ascending=True,
                )

                fig = px.bar(
                    lda_distribution,
                    x="Probability",
                    y="Topic",
                    orientation="h",
                    text="Probability",
                )

                fig.update_traces(
                    texttemplate="%{text:.1%}",
                    textposition="outside",
                )

                fig.update_layout(
                    xaxis_title="Probability",
                    yaxis_title="",
                    xaxis_tickformat=".0%",
                )

                fig = chart_layout(fig, height=420)

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={
                        "displayModeBar": False,
                    },
                )

                # ------------------------------------------------
                # NMF Distribution
                # ------------------------------------------------

                st.markdown(
                    '<div class="section-title">NMF topic weights</div>',
                    unsafe_allow_html=True,
                )

                nmf_topic_names = list(nmf_labels.values())

                nmf_distribution = pd.DataFrame(
                    {
                        "Topic": nmf_topic_names[:len(nmf_result["weights"])],
                        "Weight": nmf_result["weights"],
                    }
                ).sort_values(
                    "Weight",
                    ascending=True,
                )

                fig = px.bar(
                    nmf_distribution,
                    x="Weight",
                    y="Topic",
                    orientation="h",
                    text="Weight",
                )

                fig.update_traces(
                    texttemplate="%{text:.4f}",
                    textposition="outside",
                )

                fig.update_layout(
                    xaxis_title="Topic Weight",
                    yaxis_title="",
                )

                fig = chart_layout(fig, height=520)

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={
                        "displayModeBar": False,
                    },
                )

                # ------------------------------------------------
                # Processed Text
                # ------------------------------------------------

                with st.expander(
                    "View processed text"
                ):
                    st.code(
                        cleaned,
                        language="text",
                    )


# ============================================================
# Footer
# ============================================================

st.markdown(
    """
    <div class="footer">
        NewsTopic AI · NLP Topic Modeling · LDA + NMF
    </div>
    """,
    unsafe_allow_html=True,
)