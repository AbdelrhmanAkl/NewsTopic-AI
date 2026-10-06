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

GITHUB_URL = "https://github.com/AbdelrhmanAkl/NewsTopic-AI"
GITHUB_PROFILE = "https://github.com/AbdelrhmanAkl"

# Warm & calm palette
BG = "#fcf8f1"          # warm ivory
SURFACE = "#fffdf9"     # card surface
SIDEBAR = "#f6efe3"     # soft sand
LINE = "#eaded0"        # warm hairline
INK = "#33271f"         # espresso
MUTED = "#7b6a5d"       # warm grey-brown
HONEY = "#a9640f"       # deep honey (text / accents)
HONEY_CHART = "#d49a3c" # honey for charts
HONEY_SOFT = "#fbefd9"  # honey tint
SAGE = "#5f8a78"        # sage green
SAGE_SOFT = "#e6efe9"   # sage tint


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
# Warm & Calm UI
# ============================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,500;6..72,600;6..72,700&display=swap');

/* ---------- Global ---------- */
html, body, [class*="css"], .stApp {
    font-family: 'Figtree', -apple-system, 'Segoe UI', Arial, sans-serif;
}
.stApp { background: __BG__; }
.block-container { max-width: 1100px; padding-top: 2.2rem; padding-bottom: 4rem; }

#MainMenu, footer { visibility: hidden; }
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], .stAppDeployButton { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }

/* ---------- Typography ---------- */
h1, h2, h3, h4 { color: __INK__ !important; }
p, li, label { color: __MUTED__; }
.serif { font-family: 'Newsreader', Georgia, serif; }

.page-title {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 2.3rem; font-weight: 600; line-height: 1.15;
    color: __INK__; margin-bottom: 0.4rem;
}
.page-lead { color: __MUTED__; font-size: 1.02rem; line-height: 1.7; max-width: 680px; margin-bottom: 1.6rem; }
.section-title {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.55rem; font-weight: 600; color: __INK__; margin-bottom: 0.3rem;
}
.section-description { color: __MUTED__; font-size: 0.95rem; line-height: 1.65; margin-bottom: 1.1rem; max-width: 680px; }
.soft-divider { height: 1px; background: __LINE__; margin: 2.4rem 0; }
.muted { color: __MUTED__; font-size: 0.92rem; line-height: 1.65; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] { background: __SIDEBAR__; border-right: 1px solid __LINE__; }
section[data-testid="stSidebar"] > div { padding-top: 1.6rem; }
.brand { font-family: 'Newsreader', Georgia, serif; font-size: 1.55rem; font-weight: 700; color: __INK__; }
.brand-note { color: __MUTED__; font-size: 0.85rem; line-height: 1.6; margin-top: 0.25rem; }
.sidebar-section { color: __MUTED__; font-size: 0.8rem; font-weight: 600; margin: 1.5rem 0 0.5rem 0; }

section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 0.2rem; }
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    width: 100%; padding: 0.55rem 0.8rem; border-radius: 12px; cursor: pointer;
    transition: background 0.15s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child { display: none; }
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background: #efe5d4; }
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) { background: __HONEY_SOFT__; }
section[data-testid="stSidebar"] div[role="radiogroup"] label p { color: __INK__; font-weight: 500; font-size: 0.95rem; }
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p { color: __HONEY__; font-weight: 700; }

.sidebar-meta {
    display: flex; justify-content: space-between; padding: 0.4rem 0;
    border-bottom: 1px solid __LINE__; font-size: 0.86rem;
}
.sidebar-meta-label { color: __MUTED__; }
.sidebar-meta-value { color: __INK__; font-weight: 700; }
.sidebar-link { display: block; color: __HONEY__ !important; font-size: 0.88rem; font-weight: 600; text-decoration: none; padding: 0.25rem 0; }
.sidebar-link:hover { text-decoration: underline; }

/* ---------- Hero ---------- */
.hero { padding: 0.8rem 0 0.4rem 0; }
.hero-title {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 3.2rem; font-weight: 600; line-height: 1.08; color: __INK__;
    max-width: 760px; margin-bottom: 0.9rem;
}
.hero-subtitle { max-width: 640px; color: __MUTED__; font-size: 1.1rem; line-height: 1.75; margin-bottom: 1.4rem; }

/* ---------- Key facts strip ---------- */
.facts {
    display: grid; grid-template-columns: repeat(4, 1fr);
    background: __SURFACE__; border: 1px solid __LINE__; border-radius: 18px; overflow: hidden;
}
.fact { padding: 1.1rem 1.3rem; border-right: 1px solid __LINE__; }
.fact:last-child { border-right: none; }
.fact-value { font-family: 'Newsreader', Georgia, serif; font-size: 1.9rem; font-weight: 600; color: __HONEY__; line-height: 1.1; }
.fact-label { color: __INK__; font-size: 0.88rem; font-weight: 600; margin-top: 0.3rem; }
.fact-note { color: __MUTED__; font-size: 0.78rem; margin-top: 0.1rem; }

/* ---------- Cards (CSS grid keeps every card the same height) ---------- */
.grid { display: grid; gap: 1rem; }
.grid-2 { grid-template-columns: repeat(2, 1fr); }
.grid-3 { grid-template-columns: repeat(3, 1fr); }
.grid-5 { grid-template-columns: repeat(5, 1fr); }

.card {
    background: __SURFACE__; border: 1px solid __LINE__; border-radius: 18px;
    padding: 1.3rem 1.4rem; box-shadow: 0 1px 2px rgba(80, 55, 30, 0.04);
}
.card-icon { font-size: 1.5rem; margin-bottom: 0.5rem; }
.card-title { color: __INK__; font-size: 1.02rem; font-weight: 700; margin-bottom: 0.4rem; }
.card-text { color: __MUTED__; font-size: 0.9rem; line-height: 1.7; }

.topic-card { display: flex; gap: 0.9rem; align-items: flex-start; }
.topic-icon {
    flex: none; width: 44px; height: 44px; border-radius: 14px; background: __HONEY_SOFT__;
    display: flex; align-items: center; justify-content: center; font-size: 1.35rem;
}
.topic-icon.sage { background: __SAGE_SOFT__; }
.topic-title { color: __INK__; font-size: 1rem; font-weight: 700; margin-bottom: 0.45rem; }
.chips { display: flex; flex-wrap: wrap; gap: 0.35rem; }
.chip {
    background: #f5ede1; color: #5b4a3d; border-radius: 999px;
    padding: 0.18rem 0.65rem; font-size: 0.78rem; font-weight: 500;
}

/* ---------- Pipeline ---------- */
.step { background: __SURFACE__; border: 1px solid __LINE__; border-radius: 16px; padding: 1rem; text-align: center; }
.step-num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 28px; height: 28px; border-radius: 50%; background: __HONEY_SOFT__; color: __HONEY__;
    font-size: 0.82rem; font-weight: 700; margin-bottom: 0.5rem;
}
.step-title { color: __INK__; font-size: 0.95rem; font-weight: 700; }
.step-text { color: __MUTED__; font-size: 0.8rem; margin-top: 0.25rem; line-height: 1.5; }

/* ---------- Analyze flow ---------- */
.how { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-bottom: 1.2rem; }
.how-item {
    display: flex; align-items: center; gap: 0.5rem; background: __SURFACE__;
    border: 1px solid __LINE__; border-radius: 999px; padding: 0.35rem 0.9rem 0.35rem 0.4rem;
    color: __INK__; font-size: 0.86rem; font-weight: 500;
}
.how-item b {
    width: 22px; height: 22px; border-radius: 50%; background: __HONEY_SOFT__; color: __HONEY__;
    display: inline-flex; align-items: center; justify-content: center; font-size: 0.75rem;
}

/* ---------- Results ---------- */
.result-card { background: __SURFACE__; border: 1px solid __LINE__; border-radius: 18px; padding: 1.4rem 1.5rem; }
.result-model { color: __MUTED__; font-size: 0.86rem; font-weight: 600; }
.result-badge {
    display: inline-block; margin: 0.6rem 0 1rem 0; padding: 0.45rem 1rem; border-radius: 999px;
    background: __HONEY_SOFT__; color: __HONEY__; font-size: 1.05rem; font-weight: 700;
}
.result-badge.sage { background: __SAGE_SOFT__; color: #3f6b59; }
.result-metric-row { display: flex; justify-content: space-between; align-items: baseline; }
.result-metric-label { color: __MUTED__; font-size: 0.85rem; }
.result-metric-value { color: __INK__; font-size: 1.6rem; font-weight: 700; font-family: 'Newsreader', Georgia, serif; }
.bar { height: 8px; border-radius: 999px; background: #f1e8da; margin-top: 0.5rem; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 999px; background: __HONEY_CHART__; }
.bar-fill.sage { background: __SAGE__; }
.result-note { color: __MUTED__; font-size: 0.8rem; margin-top: 0.55rem; }

/* ---------- Inputs ---------- */
div[data-testid="stTextArea"] textarea {
    border-radius: 16px; border: 1px solid #e2d4c1; background: __SURFACE__;
    color: __INK__; font-size: 0.98rem; line-height: 1.7;
}
div[data-testid="stTextArea"] textarea:focus { border-color: __HONEY__; box-shadow: 0 0 0 3px rgba(169, 100, 15, 0.14); }
div[data-baseweb="select"] > div { border-radius: 12px; border-color: #e2d4c1; background: __SURFACE__; }

/* ---------- Buttons ---------- */
.stButton > button {
    border-radius: 12px; min-height: 44px; font-weight: 600; font-family: 'Figtree', sans-serif;
    border: 1px solid #e2d4c1; background: __SURFACE__; color: __INK__; transition: all 0.15s ease;
}
.stButton > button:hover { border-color: __HONEY__; color: __HONEY__; background: __HONEY_SOFT__; }
.stButton > button:focus-visible { outline: 3px solid rgba(169, 100, 15, 0.35); outline-offset: 2px; }
.stButton > button[kind="primary"] { background: __INK__; color: #fffaf1; border: none; }
.stButton > button[kind="primary"]:hover { background: #4a382c; color: #ffffff; box-shadow: 0 6px 18px rgba(51, 39, 31, 0.22); }

/* ---------- Tabs, metrics, expanders ---------- */
button[data-baseweb="tab"] p { color: __MUTED__; font-weight: 600; }
button[data-baseweb="tab"][aria-selected="true"] p { color: __HONEY__; }
div[data-baseweb="tab-highlight"] { background-color: __HONEY__; }
div[data-testid="stMetric"] { background: __SURFACE__; border: 1px solid __LINE__; border-radius: 16px; padding: 1rem 1.15rem; }
details { border-radius: 14px !important; border-color: __LINE__ !important; background: __SURFACE__; }

/* ---------- Footer ---------- */
.footer { text-align: center; color: #a6978a; font-size: 0.8rem; padding-top: 2.5rem; }

/* ---------- Mobile ---------- */
@media (max-width: 800px) {
    .hero-title { font-size: 2.2rem; }
    .facts { grid-template-columns: repeat(2, 1fr); }
    .fact:nth-child(2) { border-right: none; }
    .fact:nth-child(-n+2) { border-bottom: 1px solid __LINE__; }
    .grid-2, .grid-3, .grid-5 { grid-template-columns: 1fr; }
}
</style>
"""

for token, value in {
    "__BG__": BG, "__SURFACE__": SURFACE, "__SIDEBAR__": SIDEBAR, "__LINE__": LINE,
    "__INK__": INK, "__MUTED__": MUTED, "__HONEY__": HONEY, "__HONEY_CHART__": HONEY_CHART,
    "__HONEY_SOFT__": HONEY_SOFT, "__SAGE__": SAGE, "__SAGE_SOFT__": SAGE_SOFT,
}.items():
    CSS = CSS.replace(token, value)

html(CSS)


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

TOPIC_ICONS = {
    "Film, Music & Entertainment": "🎬",
    "Technology, Mobile & Gaming": "📱",
    "Politics & Government": "🏛️",
    "Sports & Football": "⚽",
    "Software, Internet & Digital Technology": "💻",
    "Business & Financial Markets": "💼",
    "Rugby & Six Nations": "🏉",
    "UK Elections & Political Parties": "🗳️",
    "Economy & Financial Markets": "📈",
    "Film & Awards": "🎞️",
    "Mobile, Broadband & Technology": "📡",
    "Law, Government & Human Rights": "⚖️",
    "Tennis & International Sports": "🎾",
    "Russian Oil & Corporate Affairs": "🛢️",
    "Football Clubs & Leagues": "⚽",
    "Music, Bands & Albums": "🎵",
}


EXAMPLE_ARTICLES = {
    "⚽ Sports": (
        "Chelsea moved clear at the top of the league after a convincing win over "
        "Arsenal at Stamford Bridge. The manager praised his players for their "
        "performance, saying the team had shown great character throughout the "
        "season. Liverpool and Manchester United both dropped points, leaving the "
        "champions with a comfortable lead ahead of the final matches of the season."
    ),
    "💼 Business": (
        "Shares in the company rose sharply after the firm reported a rise in annual "
        "profit and higher sales across its main markets. Analysts said the results "
        "were better than expected, and the bank raised its forecast for growth in "
        "the coming year. The board announced a higher dividend for shareholders "
        "despite rising oil prices and a weaker euro."
    ),
    "📱 Technology": (
        "A new mobile phone with built-in broadband access has been unveiled at a "
        "technology show. The device lets users download music and video over the "
        "network at much faster speeds. Software makers said digital services and "
        "online gaming would drive demand as more people connect to the internet "
        "from their handsets."
    ),
    "🏛️ Politics": (
        "Labour has launched its election campaign with a pledge on tax and public "
        "services. The prime minister said the government would continue its plans "
        "on policing and health, while the Tory leader attacked the party over "
        "immigration. Party strategists expect the campaign to focus on the economy "
        "in the coming weeks."
    ),
    "🎬 Entertainment": (
        "The film won the top prize at the awards ceremony, with its director and "
        "lead actor both collecting trophies. The soundtrack album, which features "
        "a well-known rock band, has also climbed the music chart. Critics called "
        "it the best film of the year, and the festival audience gave it a standing "
        "ovation."
    ),
}


# ============================================================
# State & Callbacks
# ============================================================

NAV_ITEMS = {
    "Overview": "🏠  Overview",
    "LDA Topics": "🧭  LDA Topics",
    "NMF Topics": "🔍  NMF Topics",
    "Model Comparison": "⚖️  Model Comparison",
    "Analyze Article": "✍️  Analyze Article",
}

st.session_state.setdefault("page", "Overview")
st.session_state.setdefault("article_text", "")


def set_example(text: str):
    st.session_state["article_text"] = text


def clear_text():
    st.session_state["article_text"] = ""


def go_to(page_name: str):
    st.session_state["page"] = page_name


# ============================================================
# Helper Functions
# ============================================================

def get_label(labels, topic_id):
    """Label lookup that works whether the JSON keys start at 0 or at 1."""
    offset = min(int(key) for key in labels.keys())
    return labels.get(str(topic_id + offset), f"Topic {topic_id + 1}")


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
        "label": get_label(lda_labels, topic_id),
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
        "label": get_label(nmf_labels, topic_id),
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
    """Apply a clean, warm Plotly layout."""

    fig.update_layout(
        height=height,
        template="simple_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Figtree, Arial", color="#5b4a3d"),
        margin=dict(l=20, r=40, t=20, b=20),
        xaxis=dict(gridcolor="#f1e8da", zerolinecolor="#f1e8da"),
        yaxis=dict(gridcolor="#f1e8da", zerolinecolor="#f1e8da"),
    )
    return fig


def show_chart(fig):
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
    )


def section_header(title, description=None):
    html(f'<div class="section-title">{title}</div>')
    if description:
        html(f'<div class="section-description">{description}</div>')


def page_header(title, lead):
    html(f'<div class="page-title">{title}</div>')
    html(f'<div class="page-lead">{lead}</div>')


def divider():
    html('<div class="soft-divider"></div>')


def topic_cards(topic_words, tone=""):
    cards = []
    for topic_label, words in topic_words.items():
        icon = TOPIC_ICONS.get(topic_label, "📰")
        chips = "".join(f'<span class="chip">{w}</span>' for w in words)
        cards.append(
            f'<div class="card topic-card">'
            f'<div class="topic-icon {tone}">{icon}</div>'
            f'<div><div class="topic-title">{topic_label}</div>'
            f'<div class="chips">{chips}</div></div>'
            f"</div>"
        )
    html(f'<div class="grid grid-2">{"".join(cards)}</div>')


def word_weight_chart(df, title, color):
    df = df.sort_values("Weight", ascending=True)
    fig = px.bar(
        df,
        x="Weight",
        y="Word",
        orientation="h",
        color_discrete_sequence=[color],
    )
    fig.update_traces(marker_line_width=0, opacity=0.95)
    fig.update_layout(xaxis_title=title, yaxis_title="", showlegend=False)
    return chart_layout(fig, height=400)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    html('<div class="brand">NewsTopic AI</div>')
    html(
        '<div class="brand-note">Discover the hidden themes '
        "in news articles.</div>"
    )

    html('<div class="sidebar-section">Navigate</div>')

    page = st.radio(
        "Navigation",
        list(NAV_ITEMS.keys()),
        format_func=lambda key: NAV_ITEMS[key],
        key="page",
        label_visibility="collapsed",
    )

    html('<div class="sidebar-section">The dataset</div>')

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

    html('<div class="sidebar-section">Links</div>')
    html(
        f"""
        <a class="sidebar-link" href="{GITHUB_URL}" target="_blank">Project on GitHub</a>
        <a class="sidebar-link" href="{GITHUB_PROFILE}" target="_blank">Author profile</a>
        """
    )


# ============================================================
# Overview
# ============================================================

if page == "Overview":

    html(
        """
        <div class="hero">
            <div class="hero-title">Find the themes hidden in the news</div>
            <div class="hero-subtitle">
                NewsTopic AI reads news articles and groups them by what they are
                really about, using two complementary methods: LDA and NMF.
                Paste any article and see its topic in seconds.
            </div>
        </div>
        """
    )

    cta1, cta2, _ = st.columns([1.3, 1.3, 3])
    with cta1:
        st.button(
            "Try it on an article",
            type="primary",
            use_container_width=True,
            on_click=go_to,
            args=("Analyze Article",),
        )
    with cta2:
        st.button(
            "Explore the topics",
            use_container_width=True,
            on_click=go_to,
            args=("LDA Topics",),
        )

    html('<div style="height:1.6rem"></div>')

    html(
        f"""
        <div class="facts">
            <div class="fact">
                <div class="fact-value">{TOTAL_ARTICLES:,}</div>
                <div class="fact-label">Articles analyzed</div>
                <div class="fact-note">News documents</div>
            </div>
            <div class="fact">
                <div class="fact-value">{LDA_TOPIC_COUNT}</div>
                <div class="fact-label">LDA topics</div>
                <div class="fact-note">Broad, compact themes</div>
            </div>
            <div class="fact">
                <div class="fact-value">{NMF_TOPIC_COUNT}</div>
                <div class="fact-label">NMF topics</div>
                <div class="fact-note">Finer-grained themes</div>
            </div>
            <div class="fact">
                <div class="fact-value">{LDA_COHERENCE:.4f}</div>
                <div class="fact-label">LDA coherence</div>
                <div class="fact-note">Higher means clearer topics</div>
            </div>
        </div>
        """
    )

    divider()

    # Start here
    section_header("Where would you like to start?", "Pick whichever feels most natural.")

    html(
        """
        <div class="grid grid-3">
            <div class="card">
                <div class="card-icon">✍️</div>
                <div class="card-title">Try your own article</div>
                <div class="card-text">Paste a news story or tap an example and see which topic each model picks.</div>
            </div>
            <div class="card">
                <div class="card-icon">🧭</div>
                <div class="card-title">Browse the topics</div>
                <div class="card-text">See the themes the models discovered and the words that define each one.</div>
            </div>
            <div class="card">
                <div class="card-icon">⚖️</div>
                <div class="card-title">Compare the two models</div>
                <div class="card-text">Learn how LDA and NMF differ and how closely they match the original categories.</div>
            </div>
        </div>
        """
    )
    html('<div style="height:0.6rem"></div>')

    b1, b2, b3 = st.columns(3)
    with b1:
        st.button("Analyze an article", key="start_analyze", use_container_width=True,
                  on_click=go_to, args=("Analyze Article",))
    with b2:
        st.button("Browse topics", key="start_topics", use_container_width=True,
                  on_click=go_to, args=("LDA Topics",))
    with b3:
        st.button("Compare models", key="start_compare", use_container_width=True,
                  on_click=go_to, args=("Model Comparison",))

    divider()

    # Pipeline
    section_header(
        "How an article becomes a topic",
        "Every article goes through the same five steps.",
    )

    steps = [
        ("Raw text", "The article you paste in"),
        ("Preprocessing", "Cleaning, tokenizing and lemmatizing"),
        ("Vectorization", "Bag-of-words and TF-IDF"),
        ("Modeling", "LDA and NMF"),
        ("Topics", "The dominant theme and its weights"),
    ]

    step_html = "".join(
        f'<div class="step"><div class="step-num">{i}</div>'
        f'<div class="step-title">{title}</div>'
        f'<div class="step-text">{text}</div></div>'
        for i, (title, text) in enumerate(steps, start=1)
    )
    html(f'<div class="grid grid-5">{step_html}</div>')

    divider()

    # How it works
    section_header("Two ways of finding topics")

    html(
        """
        <div class="grid grid-2">
            <div class="card">
                <div class="card-title">LDA: probabilistic modeling</div>
                <div class="card-text">
                    LDA treats each document as a mixture of hidden topics and each topic
                    as a distribution over words. The refined model gives a compact
                    six-topic view of the dataset.
                </div>
            </div>
            <div class="card">
                <div class="card-title">NMF: matrix factorization</div>
                <div class="card-text">
                    NMF breaks the TF-IDF document-term matrix into interpretable parts.
                    Its ten topics capture finer subtopics across sports, technology,
                    politics, entertainment and business.
                </div>
            </div>
        </div>
        """
    )

    divider()

    # Model summary
    section_header("Model summary")

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

    page_header(
        "LDA topics",
        "Six broad themes found by the refined LDA model. Each card lists "
        "the words that define the topic.",
    )

    topic_cards(LDA_TOPIC_WORDS)

    divider()

    section_header(
        "Look inside a topic",
        "Choose a topic to see its top words and their weights, taken straight from the trained model.",
    )

    lda_ids = list(range(lda_model.num_topics))

    selected = st.selectbox(
        "Choose a topic",
        lda_ids,
        format_func=lambda i: f"Topic {i + 1}: {get_label(lda_labels, i)}",
    )

    show_chart(
        word_weight_chart(lda_top_words(selected), "Word probability", HONEY_CHART)
    )

    st.button("Next: try an article", key="lda_to_analyze",
              on_click=go_to, args=("Analyze Article",))


# ============================================================
# NMF Topics
# ============================================================

elif page == "NMF Topics":

    page_header(
        "NMF topics",
        "Ten finer-grained themes found in the TF-IDF representation. "
        "Notice how sports split into rugby, tennis and football.",
    )

    topic_cards(NMF_TOPIC_WORDS, tone="sage")

    divider()

    section_header(
        "Look inside a topic",
        "Choose a topic to see its top terms and component weights, taken straight from the trained model.",
    )

    nmf_ids = list(range(nmf_model.components_.shape[0]))

    selected = st.selectbox(
        "Choose a topic",
        nmf_ids,
        format_func=lambda i: f"Topic {i + 1}: {get_label(nmf_labels, i)}",
    )

    show_chart(
        word_weight_chart(nmf_top_words(selected), "Component weight", SAGE)
    )

    st.button("Next: try an article", key="nmf_to_analyze",
              on_click=go_to, args=("Analyze Article",))


# ============================================================
# Model Comparison
# ============================================================

elif page == "Model Comparison":

    page_header(
        "LDA vs NMF",
        "How the two models differ in detail, and how closely each one "
        "matches the original BBC news categories.",
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
        color_discrete_map={"LDA": HONEY_CHART, "NMF": SAGE},
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        marker_line_width=0,
    )

    fig.update_layout(
        xaxis_title="",
        yaxis_title="Category alignment (%)",
        yaxis_range=[0, 100],
        showlegend=False,
        bargap=0.55,
    )

    show_chart(chart_layout(fig, height=420))

    col1, col2 = st.columns(2)

    with col1:
        st.metric("LDA coherence", f"{LDA_COHERENCE:.4f}")
        html(
            """
            <div class="muted">
            LDA gives a compact topic structure with strong semantic
            coherence and broad category coverage.
            </div>
            """
        )

    with col2:
        st.metric("NMF reconstruction error", f"{NMF_RECONSTRUCTION_ERROR:.4f}")
        html(
            """
            <div class="muted">
            NMF gives a more detailed picture, separating closely related
            areas such as football, rugby, tennis, film and music.
            </div>
            """
        )

    divider()

    section_header("What this means")

    html(
        """
        <div class="grid grid-2">
            <div class="card">
                <div class="card-title">Use LDA for the big picture</div>
                <div class="card-text">Best when you want a short list of broad categories that line up with the original news sections.</div>
            </div>
            <div class="card">
                <div class="card-title">Use NMF for the details</div>
                <div class="card-text">Best when you want specific subtopics, such as telling rugby apart from tennis.</div>
            </div>
        </div>
        """
    )
    html(
        '<div class="muted" style="margin-top:1rem">'
        "The two approaches complement each other rather than compete."
        "</div>"
    )


# ============================================================
# Analyze Article
# ============================================================

elif page == "Analyze Article":

    page_header(
        "Analyze an article",
        "See which topic each model assigns to a news article.",
    )

    html(
        """
        <div class="how">
            <div class="how-item"><b>1</b>Pick an example or paste your own</div>
            <div class="how-item"><b>2</b>Run the analysis</div>
            <div class="how-item"><b>3</b>Compare both models</div>
        </div>
        """
    )

    st.caption("Quick examples (tap one to fill the box)")

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
        placeholder="Paste an English news article here, or choose an example above...",
    )

    word_count = len(article.split())
    if word_count:
        st.caption(f"{word_count} words")

    run_col, clear_col = st.columns([4, 1])

    with run_col:
        analyze = st.button(
            "Run topic analysis",
            type="primary",
            use_container_width=True,
        )

    with clear_col:
        st.button("Clear", use_container_width=True, on_click=clear_text)

    if analyze:

        if not article.strip():

            st.warning("The box is empty. Paste an article or tap one of the examples above.")

        else:

            cleaned = preprocess_text(article)

            if not cleaned.strip():

                st.error(
                    "There is not enough usable English text in this article. "
                    "Try a longer English news story."
                )

            else:

                lda_result = predict_lda(cleaned)
                nmf_result = predict_nmf(cleaned)

                total_weight = float(nmf_result["weights"].sum())
                nmf_share = (
                    nmf_result["weight"] / total_weight if total_weight > 0 else 0.0
                )

                divider()

                section_header(
                    "Your results",
                    "Each model picked the topic that fits your article best.",
                )

                if word_count < 40:
                    st.info(
                        "This text is quite short. Articles of 50 words or more "
                        "usually give steadier results."
                    )

                html(
                    f"""
                    <div class="grid grid-2">
                        <div class="result-card">
                            <div class="result-model">LDA (probabilistic)</div>
                            <div class="result-badge">{lda_result["label"]}</div>
                            <div class="result-metric-row">
                                <span class="result-metric-label">Topic probability</span>
                                <span class="result-metric-value">{lda_result["probability"]:.1%}</span>
                            </div>
                            <div class="bar"><div class="bar-fill" style="width:{lda_result["probability"] * 100:.1f}%"></div></div>
                        </div>
                        <div class="result-card">
                            <div class="result-model">NMF (matrix factorization)</div>
                            <div class="result-badge sage">{nmf_result["label"]}</div>
                            <div class="result-metric-row">
                                <span class="result-metric-label">Share of total topic weight</span>
                                <span class="result-metric-value">{nmf_share:.1%}</span>
                            </div>
                            <div class="bar"><div class="bar-fill sage" style="width:{nmf_share * 100:.1f}%"></div></div>
                            <div class="result-note">Raw topic weight: {nmf_result["weight"]:.4f}</div>
                        </div>
                    </div>
                    """
                )

                html('<div style="height:1.2rem"></div>')

                tab_lda, tab_nmf = st.tabs(["LDA distribution", "NMF weights"])

                with tab_lda:
                    lda_distribution = pd.DataFrame(
                        [
                            {
                                "Topic": get_label(lda_labels, topic_id),
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
                        color_discrete_sequence=[HONEY_CHART],
                    )

                    fig.update_traces(
                        texttemplate="%{text:.1%}",
                        textposition="outside",
                        marker_line_width=0,
                        opacity=0.95,
                    )

                    fig.update_layout(
                        xaxis_title="Probability",
                        yaxis_title="",
                        xaxis_tickformat=".0%",
                        showlegend=False,
                    )

                    show_chart(chart_layout(fig, height=420))

                with tab_nmf:
                    nmf_topic_names = [
                        get_label(nmf_labels, i)
                        for i in range(len(nmf_result["weights"]))
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
                        color_discrete_sequence=[SAGE],
                    )

                    fig.update_traces(
                        texttemplate="%{text:.4f}",
                        textposition="outside",
                        marker_line_width=0,
                        opacity=0.95,
                    )

                    fig.update_layout(
                        xaxis_title="Topic weight",
                        yaxis_title="",
                        showlegend=False,
                    )

                    show_chart(chart_layout(fig, height=520))

                with st.expander("How to read these results"):
                    st.markdown(
                        "- **LDA** gives probabilities that add up to 100% across its six topics.\n"
                        "- **NMF** gives weights over its ten topics. We show the winning "
                        "topic's share of the total so it is easier to compare.\n"
                        "- If the two models name different topics, that is normal: "
                        "NMF is more specific (for example *Football Clubs* vs. LDA's *Sports*)."
                    )

                with st.expander("See the processed text"):
                    st.code(cleaned, language="text")


# ============================================================
# Footer
# ============================================================

html(
    """
    <div class="footer">
        NewsTopic AI · Topic modeling with LDA and NMF
    </div>
    """
)
