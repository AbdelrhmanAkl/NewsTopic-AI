import json
from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


def load_nmf_model():
    """Load the trained NMF model."""
    return joblib.load(MODELS_DIR / "final_nmf_model.joblib")


def load_tfidf_vectorizer():
    """Load the TF-IDF vectorizer used during NMF training."""
    return joblib.load(MODELS_DIR / "refined_tfidf_vectorizer.joblib")


def load_nmf_topic_labels():
    """Load human-readable NMF topic labels."""
    with open(MODELS_DIR / "nmf_topic_labels.json", "r", encoding="utf-8") as file:
        return json.load(file)


def predict_nmf(text: str):
    """
    Predict the dominant topic for a preprocessed document.

    Returns:
        dominant_topic: Zero-based topic index.
        topic_label: Human-readable topic name.
        weight: NMF topic weight.
        topic_distribution: All topic weights.
    """
    model = load_nmf_model()
    vectorizer = load_tfidf_vectorizer()
    labels = load_nmf_topic_labels()

    tfidf_matrix = vectorizer.transform([text])

    topic_weights = model.transform(tfidf_matrix)[0]

    dominant_topic = int(topic_weights.argmax())
    weight = float(topic_weights[dominant_topic])

    topic_label = labels[str(dominant_topic)]

    return {
        "dominant_topic": dominant_topic,
        "topic_label": topic_label,
        "weight": weight,
        "topic_distribution": {
            int(topic_id): float(topic_weight)
            for topic_id, topic_weight in enumerate(topic_weights)
        },
    }