import os
import sys
import joblib
import numpy as np

# Add parent directory to path so we can import
# the NLPPreprocessor class used inside the pipelines.
# Without this, joblib cannot unpickle the models.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."
))
from nlp_preprocessor import NLPPreprocessor  # noqa: E402, F401


# -------------------------------
# MODEL PATH HELPER
# -------------------------------
def get_model_path(model_name):
    """
    Maps model names to .pkl files.
    """

    model_filenames = {
        "KNN": "knn_model.pkl",
        "SVM": "svm_model.pkl",
        "AdaBoost": "adaboost_model.pkl",
        "XGBoost": "xgboost_model.pkl",
        "Gradient Boosting": "gradient_boosting_model.pkl"
    }

    filename = model_filenames.get(model_name)

    if not filename:
        raise ValueError(
            f"Model '{model_name}' not found."
        )

    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    filepath = os.path.abspath(
        os.path.join(
            base_dir,
            "..",
            "models",
            filename
        )
    )

    return filepath


# -------------------------------
# LOAD MODEL
# -------------------------------
def load_model(model_name):
    """
    Loads sklearn pipeline model.
    """

    filepath = get_model_path(model_name)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Missing model file: {filepath}"
        )

    model = joblib.load(filepath)

    return model


# -------------------------------
# LABEL MAPPING
# -------------------------------
def map_prediction_label(raw_prediction):
    """
    Robust label mapping.
    Handles:
    0/1
    Fake/Real
    False/True
    """

    pred_str = str(
        raw_prediction
    ).strip().upper()

    if pred_str in [
        "1",
        "1.0",
        "REAL",
        "TRUE"
    ]:
        return "Real"

    elif pred_str in [
        "0",
        "0.0",
        "FAKE",
        "FALSE"
    ]:
        return "Fake"

    return str(raw_prediction)


# -------------------------------
# GET CONFIDENCE
# -------------------------------
def get_confidence(
    model,
    text,
    raw_prediction
):
    """
    Extract confidence safely
    using predict_proba.
    """

    confidence = 100.0

    if hasattr(model, "predict_proba"):

        probabilities = (
            model.predict_proba([text])[0]
        )

        if hasattr(model, "classes_"):

            classes_list = list(
                model.classes_
            )

            if raw_prediction in classes_list:

                class_index = (
                    classes_list.index(
                        raw_prediction
                    )
                )

                confidence = float(
                    probabilities[
                        class_index
                    ]
                ) * 100

            else:
                confidence = (
                    float(
                        np.max(probabilities)
                    ) * 100
                )

        else:
            confidence = (
                float(
                    np.max(probabilities)
                ) * 100
            )

    return confidence


# -------------------------------
# MAIN PREDICTION ENGINE
# -------------------------------
def predict_all_models(text):
    """
    Generator function.

    Yields results progressively
    for Streamlit animation.
    """

    model_order = [
        "KNN",
        "SVM",
        "AdaBoost",
        "XGBoost",
        "Gradient Boosting"
    ]

    results = []

    for model_name in model_order:

        try:
            # Load model
            model = load_model(
                model_name
            )

            # Predict
            raw_prediction = (
                model.predict([text])[0]
            )

            # Label mapping
            label = (
                map_prediction_label(
                    raw_prediction
                )
            )

            # Confidence
            confidence = (
                get_confidence(
                    model,
                    text,
                    raw_prediction
                )
            )

            # Save result
            results.append({
                "model": model_name,
                "label": label,
                "probability": confidence
            })

            # ----------------
            # MAJORITY VOTE
            # ----------------
            real_votes = sum(
                1
                for r in results
                if r["label"] == "Real"
            )

            fake_votes = sum(
                1
                for r in results
                if r["label"] == "Fake"
            )

            final_label = (
                "Real"
                if real_votes >= fake_votes
                else "Fake"
            )

            # ----------------
            # AVG CONFIDENCE
            # ----------------
            avg_probability = (
                sum(
                    r["probability"]
                    for r in results
                )
                / len(results)
            )

            data = {
                "results": results.copy(),
                "average_probability":
                    avg_probability,
                "final_label":
                    final_label
            }

            print(
                "Yielding:",
                data
            )

            # IMPORTANT:
            # progressive updates
            yield data

        except Exception as e:
            print(
                f"Error in "
                f"{model_name}: {e}"
            )