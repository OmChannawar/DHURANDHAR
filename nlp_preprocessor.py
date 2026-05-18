"""
==========================================================
DHURANDHAR — Shared NLP Preprocessing Module
==========================================================

This module contains the NLPPreprocessor class and
preprocess_text function used by both:
  - train_models.py (training)
  - ml_backend.py (inference)

IMPORTANT: This file must be importable at both
training time and inference time so that joblib
can correctly unpickle the pipeline models.
==========================================================
"""

import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

# Download NLTK data (only downloads if not already present)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# NLP Setup
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# ==========================================================
# NLP PREPROCESSING FUNCTION
# ==========================================================

def preprocess_text(text):
    """
    NLP preprocessing pipeline:
    lowercase → remove punctuation → tokenize →
    remove stopwords → lemmatize → rejoin
    """
    text = str(text).lower()

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    words = text.split()

    cleaned_words = []

    for word in words:
        if word.isalpha() and word not in stop_words:
            word = lemmatizer.lemmatize(word)
            cleaned_words.append(word)

    return " ".join(cleaned_words)


# ==========================================================
# CUSTOM NLP TRANSFORMER (for sklearn Pipeline)
# ==========================================================

class NLPPreprocessor(BaseEstimator, TransformerMixin):
    """
    Custom sklearn transformer that applies
    NLP preprocessing to raw text input.

    Used inside sklearn Pipeline so that the
    entire model can accept raw text directly.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [
            preprocess_text(text)
            for text in X
        ]
