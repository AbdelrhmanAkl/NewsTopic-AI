import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


NEWS_STOPWORDS = {
    "said",
    "say",
    "says",
    "would",
    "could",
    "also",
    "one",
    "two",
    "three",
    "year",
    "years",
    "time",
    "new",
    "like",
    "get",
    "got",
    "make",
    "made",
    "way",
    "back",
    "first",
    "last",
    "people",
}


def download_nltk_resources():
    """Download the NLTK resources required by the preprocessing pipeline."""
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
    ]

    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)


def preprocess_text(text: str) -> str:
    """
    Apply the same preprocessing pipeline used during model training.

    Steps:
    1. Ensure required NLTK resources are available.
    2. Convert text to lowercase.
    3. Remove URLs.
    4. Keep alphabetic characters only.
    5. Tokenize.
    6. Remove standard English stopwords.
    7. Remove tokens with length <= 2.
    8. Lemmatize tokens.
    9. Remove domain-specific news stopwords.
    """
    if not isinstance(text, str):
        return ""

    download_nltk_resources()

    text = text.lower()

    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    tokens = text.split()

    english_stopwords = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    tokens = [
        token
        for token in tokens
        if token not in english_stopwords and len(token) > 2
    ]

    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
    ]

    tokens = [
        token
        for token in tokens
        if token not in NEWS_STOPWORDS
    ]

    return " ".join(tokens)


def tokenize_for_lda(text: str) -> list[str]:
    """Convert preprocessed text into tokens for the LDA model."""
    cleaned_text = preprocess_text(text)
    return cleaned_text.split()