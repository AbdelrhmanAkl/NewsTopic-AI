import json
import pickle
from pathlib import Path

from gensim.models import LdaModel


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


def load_lda_model():
    """Load the trained LDA model."""
    return LdaModel.load(str(MODELS_DIR / "final_lda_model"))


def load_lda_dictionary():
    """Load the dictionary used during LDA training."""
    with open(MODELS_DIR / "refined_dictionary.pkl", "rb") as file:
        return pickle.load(file)


def load_lda_topic_labels():
    """Load human-readable LDA topic labels."""
    with open(MODELS_DIR / "lda_topic_labels.json", "r", encoding="utf-8") as file:
        return json.load(file)


def predict_lda(text: str):
    """
    Predict the dominant topic for a preprocessed document.

    Returns:
        dominant_topic: Zero-based topic index.
        topic_label: Human-readable topic name.
        probability: LDA topic probability.
        topic_distribution: All topic probabilities.
    """
    model = load_lda_model()
    dictionary = load_lda_dictionary()
    labels = load_lda_topic_labels()

    bow = dictionary.doc2bow(text.split())

    topic_distribution = model.get_document_topics(
        bow,
        minimum_probability=0,
    )

    dominant_topic, probability = max(
        topic_distribution,
        key=lambda item: item[1],
    )

    topic_label = labels[str(dominant_topic)]

    return {
        "dominant_topic": dominant_topic,
        "topic_label": topic_label,
        "probability": float(probability),
        "topic_distribution": {
            int(topic_id): float(topic_probability)
            for topic_id, topic_probability in topic_distribution
        },
    }