import os
import joblib
import numpy as np

def get_model_path(model_name):
    """
    Helper function to map the model name to the .pkl file path.
    """
    model_filenames = {
        "Logistic Regression": "logistic_regression_model.pkl",
        "KNN": "knn_model.pkl",
        "SVM": "svm_model.pkl",
        "Decision Tree": "decision_tree_model.pkl",
        "Naive Bayes": "naive_bayes_model.pkl",
        "Random Forest": "random_forest_model.pkl",
        "AdaBoost": "adaboost_model.pkl",
        "Gradient Boosting": "gradient_boosting_model.pkl",
        "XGBoost": "xgboost_model.pkl"
    }
    filename = model_filenames.get(model_name)
    if not filename:
        return None
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base_dir, "..", "models", filename))

def load_model(model_name):
    """
    Loads the pipeline model via joblib.
    """
    filepath = get_model_path(model_name)
    if not filepath or not os.path.exists(filepath):
        return None
    return joblib.load(filepath)

def format_label(pred):
    """
    Safely normalizes outputs into clean 'Real' or 'Fake' strings,
    independent of how models were initially label-encoded.
    """
    val = str(pred).strip().upper()
    if val in ['1', '1.0', 'REAL', 'TRUE']:
        return "Real"
    elif val in ['0', '0.0', 'FAKE', 'FALSE']:
        return "Fake"
    return str(pred).title()

def predict_all_models(text):
    """
    Generator that evaluates the text against all models sequentially.
    Yields a dictionary with the accumulated results so far, enabling 
    live, real-time UI updates as each model finishes processing.
    """
    model_names = [
        "Logistic Regression", "KNN", "SVM", "Decision Tree", 
        "Naive Bayes", "Random Forest", "AdaBoost", 
        "Gradient Boosting", "XGBoost"
    ]
    
    results = []
    
    for model_name in model_names:
        try:
            model = load_model(model_name)
            if not model:
                continue
            
            # 1. Pipeline inference
            pred = model.predict([text])[0]
            
            # 2. Probability extraction mapping via classes_
            confidence = 100.0
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba([text])[0]
                if hasattr(model, "classes_"):
                    classes = list(model.classes_)
                    if pred in classes:
                        idx = classes.index(pred)
                        confidence = float(proba[idx]) * 100
                    else:
                        confidence = float(np.max(proba)) * 100
                else:
                    confidence = float(np.max(proba)) * 100
            
            label_str = format_label(pred)
            
            results.append({
                "model": model_name,
                "label": label_str,
                "probability": confidence
            })
            
            # 3. Calculate running metrics
            avg_prob = sum(r["probability"] for r in results) / len(results)
            real_count = sum(1 for r in results if r["label"] == "Real")
            fake_count = sum(1 for r in results if r["label"] == "Fake")
            
            # 4. Ensemble Majority Vote
            if real_count > fake_count:
                final_label = "Real"
            elif fake_count > real_count:
                final_label = "Fake"
            else:
                final_label = "Tie"
                
            # Yield structure exactly as requested
            yield {
                "results": results,
                "average_probability": avg_prob,
                "final_label": final_label
            }
            
        except Exception as e:
            print(f"Skipping {model_name} due to error: {e}")
            continue