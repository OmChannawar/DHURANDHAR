import sys, os
sys.path.insert(0, '.')
from nlp_preprocessor import NLPPreprocessor
import joblib

text = """As Donald Trump grew increasingly frustrated with diplomatic efforts to end the war with Iran, administration officials were closely watching whether the president's trip to China would yield a significant breakthrough. But Trump landed stateside Friday with seemingly no progress to report. Speaking to reporters on his journey back to Washington, the US president claimed Chinese leader Xi Jinping said he would like the Strait of Hormuz to be reopened and that he agrees Iran should not develop a nuclear weapon. But those were statements China had made previously."""

models = ['knn_model', 'svm_model', 'adaboost_model', 'xgboost_model', 'gradient_boosting_model']
for m in models:
    model = joblib.load(f'models/{m}.pkl')
    pred = model.predict([text])[0]
    label = 'REAL' if pred == 1 else 'FAKE'
    
    # Get confidence
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba([text])[0]
        conf = max(proba) * 100
    else:
        conf = 100.0
    
    print(f'{m}: {label} ({conf:.1f}%)')
