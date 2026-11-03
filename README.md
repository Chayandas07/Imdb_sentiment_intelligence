# Imdb_sentiment_intelligence
An end to end NLP and Machine Learning project for IMDb movie review sentiment classification using TF - IDF, Naive Bayes, Logistic Regression, Linear SVM, hyperparameter optimization, explainable AI, and interactive Gradio deployment.

# 🎬 IMDb Sentiment Intelligence

### An End-to-End NLP & Machine Learning Project for Movie Review Sentiment Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![NLP](https://img.shields.io/badge/Domain-NLP-purple.svg)](https://en.wikipedia.org/wiki/Natural_language_processing)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)](https://scikit-learn.org/)
[![Google Colab](https://img.shields.io/badge/Platform-Google%20Colab-yellow.svg)](https://colab.research.google.com/)
[![Gradio](https://img.shields.io/badge/Interface-Gradio-red.svg)](https://www.gradio.app/)
[![Status](https://img.shields.io/badge/Status-Implemented-success.svg)]()

---

## 📌 Overview

**IMDb Sentiment Intelligence** is an end-to-end Natural Language Processing (NLP) and Machine Learning project that automatically classifies movie reviews into two sentiment categories: **Positive** and **Negative**.

The project uses the IMDb 50K Movie Reviews dataset and implements advanced text preprocessing, TF-IDF feature extraction, multiple machine learning algorithms, cross-validation, and hyperparameter optimization.

It also incorporates model explainability, comprehensive performance evaluation, model persistence, and an interactive Gradio application for real-time sentiment prediction.

The primary objective is to demonstrate a complete machine learning workflow, from raw text processing to model development, evaluation, and deployment.

## 🎯 Project Objectives

- Develop an automated movie review sentiment classification system.
- Implement effective NLP preprocessing techniques.
- Extract meaningful textual features using TF-IDF.
- Compare multiple supervised machine learning algorithms.
- Optimize model performance using cross-validation and GridSearchCV.
- Evaluate classification performance using multiple ML metrics.
- Identify influential words and phrases using model coefficients.
- Build an interactive application for real-time sentiment prediction.

## 📊 Dataset

**IMDb 50K Movie Reviews Dataset**

The dataset contains approximately 50,000 movie reviews labeled as positive or negative.

**Dataset Source:** [IMDb 50K Movie Reviews - Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

| Attribute | Description |
|---|---|
| Dataset | IMDb 50K Movie Reviews |
| Total Reviews | 50,000 |
| Input Feature | Movie Review Text |
| Target Variable | Sentiment |
| Positive Class | 1 |
| Negative Class | 0 |
| Problem Type | Binary Classification |

### Dataset Splitting

- Training Dataset: 80% (approximately 40,000 reviews)
- Testing Dataset: 20% (approximately 10,000 reviews)

Stratified sampling is used to maintain class distribution across training and testing datasets.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | NumPy, Pandas |
| Data Visualization | Matplotlib, Seaborn |
| NLP | Regex, TF-IDF |
| Machine Learning | Scikit-learn |
| Classification | Naive Bayes, Logistic Regression, Linear SVM |
| Optimization | GridSearchCV |
| Evaluation | Accuracy, Precision, Recall, F1-score, ROC-AUC |
| Explainable AI | Feature Coefficient Analysis |
| Model Persistence | Joblib |
| Interactive Interface | Gradio |
| Development Environment | Google Colab |

## 🧠 Machine Learning Workflow

The project follows a structured end-to-end machine learning pipeline:

1. **Data Acquisition:** Download and load the IMDb 50K dataset.
2. **Data Validation:** Handle missing values, duplicate reviews, and invalid labels.
3. **Exploratory Data Analysis:** Analyze sentiment distribution and review-length patterns.
4. **Text Preprocessing:** Remove HTML tags, URLs, and unnecessary whitespace.
5. **Data Splitting:** Perform stratified training and testing splits.
6. **Feature Engineering:** Extract textual features using TF-IDF with unigram and bigram representations.
7. **Model Development:** Train and compare multiple machine learning classifiers.
8. **Cross-Validation:** Evaluate model stability using Stratified K-Fold cross-validation.
9. **Hyperparameter Optimization:** Optimize Linear SVM using GridSearchCV.
10. **Model Evaluation:** Evaluate the final classifier using multiple performance metrics.
11. **Explainable AI:** Identify influential positive and negative textual features.
12. **Deployment:** Save the trained model and develop an interactive Gradio application.

## 🤖 Machine Learning Algorithms

Three supervised machine learning algorithms are implemented and compared.

### 1. Multinomial Naive Bayes

- Probabilistic classification algorithm.
- Suitable for text classification tasks.
- Provides a computationally efficient baseline.

### 2. Logistic Regression

- Linear classification algorithm.
- Estimates probabilities for binary classification.
- Provides interpretable feature coefficients.

### 3. Linear Support Vector Machine (SVM)

- Margin-based classification algorithm.
- Effective for high-dimensional sparse textual features.
- Uses regularization to control model complexity.

### Hyperparameter Optimization

GridSearchCV is used to optimize the Linear SVM regularization parameter.

**Search Space:**

```python
C = [0.5, 1.0, 2.0, 3.0]
```

The best configuration is selected based on cross-validation F1-score.

## 📈 Model Evaluation

The final selected model is evaluated using:

| Metric | Purpose |
|---|---|
| Accuracy | Measures overall classification correctness |
| Precision | Measures the correctness of positive predictions |
| Recall | Measures the ability to identify actual positive reviews |
| F1-score | Balances precision and recall |
| ROC-AUC | Measures the ability to distinguish sentiment classes |

### Evaluation Visualizations

- Confusion Matrix
- ROC Curve
- Model Comparison Chart
- Feature Importance Visualization

Actual test-set performance metrics are generated by executing the notebook.

## 🔍 Explainable AI (XAI)

The project incorporates feature-level explainability for linear classification models.

By analyzing learned model coefficients, the system identifies:

- Words and phrases associated with positive sentiment.
- Words and phrases associated with negative sentiment.
- The relative influence of textual features on the model's decision boundary.

This provides insights into the model's learned patterns and improves interpretability.

## 🚀 Interactive Sentiment Prediction

An interactive Gradio application is developed to demonstrate real-time sentiment classification.

### Application Features

- Enter any movie review.
- Analyze the textual content.
- Predict positive or negative sentiment.
- Display the model's decision score.
- Interact with the trained machine learning pipeline.

The application provides a simple interface for testing the trained model without manually executing prediction code.

## 📂 Project Structure

```text
IMDb-Sentiment-Intelligence/
│
├── IMDb_Sentiment_Intelligence.ipynb
│
├── artifacts/
│   ├── imdb_sentiment_model.joblib
│   ├── model_comparison.csv
│   ├── final_metrics.csv
│   ├── top_features.csv
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
│
├── README.md
│
└── .gitignore
```

*The artifacts directory contains files generated during model training and evaluation.*

## ⚙️ Installation & Execution

### Option 1: Google Colab (Recommended)

1. Open the project notebook.
2. Upload it to Google Colab.
3. Execute the notebook cells sequentially.
4. Download the IMDb dataset using KaggleHub.
5. Perform preprocessing and feature extraction.
6. Train and compare machine learning models.
7. Optimize the selected model.
8. Evaluate the final model.
9. Launch the interactive Gradio application.

### Option 2: Run Locally

Clone the repository:

```bash
git clone https://github.com/Chayandas07/Imdb_sentiment_intelligenc.git
```

Navigate to the project directory:

```bash
cd Imdb_sentiment_intelligenc
```

Install the dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn kagglehub joblib gradio
```

Open the notebook:

```bash
jupyter notebook
```

Execute the notebook cells sequentially.

## 💾 Model Artifacts

The project generates reusable model and evaluation files.

| File | Description |
|---|---|
| `imdb_sentiment_model.joblib` | Complete trained TF-IDF and classification pipeline |
| `model_comparison.csv` | Comparative machine learning performance |
| `final_metrics.csv` | Final test evaluation metrics |
| `top_features.csv` | Influential positive and negative features |
| `confusion_matrix.png` | Classification error visualization |
| `roc_curve.png` | ROC performance visualization |
| `feature_importance.png` | Feature coefficient visualization |

## 🔮 Future Enhancements

Potential improvements to extend this project:

- **Word2Vec:** Explore semantic word embeddings.
- **DistilBERT:** Fine-tune transformer-based models for contextual sentiment understanding.
- **BERT:** Experiment with advanced contextual NLP architectures.
- **FastAPI:** Develop a REST API for model inference.
- **Docker:** Containerize the application for reproducible deployment.
- **MLflow:** Track experiments, model versions, and evaluation metrics.
- **SHAP and LIME:** Implement advanced explainability techniques.
- **Model Monitoring:** Track prediction patterns and data drift.

## 🎓 Learning Outcomes

This project demonstrates practical knowledge of:

- End-to-end machine learning development.
- Natural Language Processing and text preprocessing.
- TF-IDF and n-gram feature engineering.
- Supervised machine learning algorithms.
- Cross-validation and hyperparameter optimization.
- Model evaluation and error analysis.
- Explainable machine learning.
- Model persistence and interactive deployment.
