## 👨‍💻 Team Members

| Member | Name | Primary Responsibility |
|---------|------|-------------------------|
| Member 1 | Om C | Preprocessing, Classification & Ensemble |
| Member 2 | Vishal | Classification, Ensemble & Evaluation |
| Member 3 | Om M | Research, Deployment & Comparative Analysis |

---

## 🤝 Team Contribution & Work Distribution

To ensure **equal participation and technical contribution**, the project work was divided evenly among all three team members.

Each member contributed to:

- Data preprocessing
- Model implementation
- Performance evaluation
- Documentation
- Presentation preparation
- Poster creation
- Viva preparation
- GitHub maintenance

---

### 👨‍💻 Member 1 — Om Channawar

#### **Project Contributions**

##### **Dataset Pipeline & NLP Preprocessing**
* **Engineered** the data pipeline by merging the `Fake.csv` and `True.csv` datasets, handling missing values, and applying label encoding.
* **Implemented** a comprehensive NLP text cleaning pipeline, executing lowercase conversion, punctuation and stopword removal, tokenization, and lemmatization.
* **Transformed** raw textual data into numerical features using TF-IDF Vectorization.

##### **Machine Learning Modeling & Evaluation**
* **Developed and trained** three baseline Machine Learning models: Logistic Regression, K-Nearest Neighbors (KNN), and Random Forest.
* **Evaluated** model performance by analyzing Confusion Matrices, Accuracy, Precision, Recall, and F1-Scores.

---

### 👨‍💻 Member 2 — Vishal Shende

#### **Project Contributions**

##### **Advanced Machine Learning & Ensemble Modeling**
* **Built and optimized** core classification models using Support Vector Machine (SVM), Decision Tree, and Naïve Bayes algorithms.
* **Enhanced** overall prediction accuracy by implementing ensemble techniques, specifically AdaBoost and Gradient Boosting.

##### **Model Evaluation & Diagnostic Visualizations**
* **Conducted** ROC Curve Analysis to evaluate classifier thresholds and trade-offs.
* **Profiled** computational efficiency by analyzing and comparing model training times.
* **Generated** performance charts and conducted Feature Importance Analysis to identify the key textual indicators of fake news.

---

### 👨‍💻 Member 3 — Om Mapari

#### **Project Contributions**

##### **Research Framework & Literature Review**
* **Analyzed** key methodologies from a contemporary research paper on *Fake News Detection using Machine Learning and NLP*.
* **Formulated** the project’s foundational framework, including the Aim, Objectives, Problem Statement, and overarching Methodology.

##### **Advanced Modeling, Deployment & System Synthesis**
* **Integrated** XGBoost into the ensemble pipeline to maximize predictive performance.
* **Designed and deployed** a fully functional, interactive user interface using Streamlit that accepts live user news inputs and outputs real-time Fake/Real predictions alongside a model confidence score.

##### **Comparative Analysis & Diagnostics**
* **Constructed** the final model benchmarking table to evaluate all algorithms side-by-side.
* **Diagnosed** system boundaries by performing Overfitting vs. Underfitting diagnostics and evaluating computational complexity to recommend the optimal final deployment model.

---

## 📂 Project Structure

```text
FAKE-NEWS-DETECTION/
│
├── dataset/
│   ├── Fake.csv
│   ├── True.csv
│   └── processed_fake_news_dataset.csv
│
├── deployment/
│   ├── app.py
│   ├── ml_backend.py
│   └── requirements.txt
│
├── models/
│   ├── knn_model.pkl
│   ├── svm_model.pkl
│   ├── adaboost_model.pkl
│   └── xgboost_model.pkl
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_knn.ipynb
│   ├── 03_svm.ipynb
│   ├── 04_adaboost.ipynb
│   └── 05_xgboost.ipynb
│
├── poster/
├── presentation/
├── screenshots/
├── research_paper/
│
└── README.md
```

---

## 🤝 Collaborative Contributions

All team members actively contributed to:

### Documentation
- README.md
- Code comments
- Result interpretation

### Presentation
- PowerPoint preparation
- Visualizations
- Explanation flow

### Poster Design
- Problem statement
- Methodology
- Results
- Conclusions

### GitHub Repository
All members contributed through regular commits and collaborative development.

---

## 🏆 Equal Participation Statement

This project was developed through **equal contribution and collaborative effort** by all three team members. Every member actively participated in implementation, experimentation, analysis, documentation, deployment, and viva preparation.
