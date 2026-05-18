import os
import sys
import time
import warnings

import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.calibration import CalibratedClassifierCV

# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from xgboost import XGBClassifier

# Shared NLP module (must be importable at inference too)
from nlp_preprocessor import NLPPreprocessor

warnings.filterwarnings("ignore")


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")

FAKE_CSV = os.path.join(DATASET_DIR, "Fake.csv")
TRUE_CSV = os.path.join(DATASET_DIR, "True.csv")


# ==========================================================
# STEP 1: LOAD & PREPARE DATA
# ==========================================================

def load_data():
    print("\n" + "=" * 60)
    print("STEP 1: LOADING DATASETS")
    print("=" * 60)

    fake_df = pd.read_csv(FAKE_CSV)
    true_df = pd.read_csv(TRUE_CSV)

    print(f"  Fake news articles: {len(fake_df)}")
    print(f"  Real news articles: {len(true_df)}")

    # Add labels
    fake_df["label"] = 0  # Fake
    true_df["label"] = 1  # Real

    # Merge
    df = pd.concat([fake_df, true_df], axis=0)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # ----------------------------------------------------------
    # CRITICAL: Strip Reuters datelines from True articles
    # ----------------------------------------------------------
    # All "True" news articles start with "CITY (Reuters) - ..."
    # This is a massive data leakage: models learn to associate
    # the "(Reuters)" tag with real news, and classify any
    # non-Reuters article as fake.
    #
    # Pattern: "WASHINGTON (Reuters) - " or "LONDON (Reuters) - "
    # We strip everything before and including " - " on the
    # first occurrence, but only if "(Reuters)" is present.
    # ----------------------------------------------------------
    import re

    def strip_reuters_dateline(text):
        text = str(text)
        # Remove "CITY (Reuters) - " prefix
        text = re.sub(
            r'^[A-Z\s/]+ \(Reuters\)\s*-\s*',
            '',
            text
        )
        return text

    df["text"] = df["text"].apply(strip_reuters_dateline)

    # Combine title + text into content
    # NOTE: subject is excluded intentionally.
    # The subject column has completely different values
    # between Fake and True datasets, creating data leakage.
    df["content"] = (
        df["title"].astype(str)
        + " "
        + df["text"].astype(str)
    )

    # Drop missing
    df = df.dropna(subset=["content", "label"])

    # Drop duplicates
    df = df.drop_duplicates(subset=["content"])

    print(f"  Total articles after cleaning: {len(df)}")
    print(f"  Label distribution:\n{df['label'].value_counts().to_string()}")

    return df


# ==========================================================
# STEP 2: DEFINE MODEL CONFIGS
# ==========================================================

def get_model_configs():
    """
    Returns list of (filename, model_name, classifier) tuples.

    All pipelines follow: NLPPreprocessor → TfidfVectorizer → Classifier

    Note: LinearSVC doesn't natively support predict_proba,
    so we wrap it with CalibratedClassifierCV.
    """

    return [
        (
            "knn_model.pkl",
            "KNN",
            KNeighborsClassifier(
                n_neighbors=5,
                weights="distance"
            )
        ),
        (
            "svm_model.pkl",
            "SVM",
            CalibratedClassifierCV(
                LinearSVC(random_state=42, max_iter=5000),
                cv=3
            )
        ),
        (
            "adaboost_model.pkl",
            "AdaBoost",
            AdaBoostClassifier(
                n_estimators=200,
                learning_rate=0.1,
                random_state=42
            )
        ),
        (
            "xgboost_model.pkl",
            "XGBoost",
            XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                reg_alpha=1.0,
                reg_lambda=5.0,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss",
                verbosity=0
            )
        ),
        (
            "gradient_boosting_model.pkl",
            "Gradient Boosting",
            GradientBoostingClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
        ),
    ]


# ==========================================================
# STEP 3: TRAIN & SAVE ALL MODELS
# ==========================================================

def train_and_save_all(df):
    print("\n" + "=" * 60)
    print("STEP 2: TRAINING MODEL PIPELINES")
    print("=" * 60)

    # Features & Labels
    X_raw = df["content"]
    y = df["label"]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples:  {len(X_test)}")

    # Create models directory
    os.makedirs(MODELS_DIR, exist_ok=True)

    configs = get_model_configs()

    for filename, model_name, classifier in configs:

        print(f"\n{'-' * 50}")
        print(f"  Training: {model_name}")
        print(f"{'-' * 50}")

        # Build Pipeline
        pipeline = Pipeline([
            (
                "nlp",
                NLPPreprocessor()
            ),
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    max_df=0.7,
                    min_df=2
                )
            ),
            (
                "classifier",
                classifier
            )
        ])

        # Train
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Evaluate
        predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        print(f"  Accuracy:   {accuracy:.4f}")
        print(f"  Train time: {train_time:.1f}s")

        # Save
        filepath = os.path.join(MODELS_DIR, filename)
        joblib.dump(pipeline, filepath)
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  Saved:      {filepath} ({file_size_mb:.1f} MB)")

    # ==========================================================
    # SUMMARY
    # ==========================================================

    print("\n" + "=" * 60)
    print("ALL MODELS TRAINED & SAVED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nModels directory: {MODELS_DIR}")
    print("Files saved:")
    for f in os.listdir(MODELS_DIR):
        fpath = os.path.join(MODELS_DIR, f)
        size = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  [OK] {f} ({size:.1f} MB)")


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    total_start = time.time()

    df = load_data()
    train_and_save_all(df)

    total_time = time.time() - total_start
    print(f"\nTotal execution time: {total_time:.1f}s")
