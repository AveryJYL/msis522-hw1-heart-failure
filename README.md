# ❤️ Heart Failure Prediction - MSIS 522 HW1

An end-to-end data science workflow for predicting heart disease using clinical features.

## 🔗 Live Demo

[**Streamlit App**](https://msis522-hw1-heart-failure-gjfteetwerufsrd8tkvd63.streamlit.app/)

## 📊 Overview

This project implements the complete data science pipeline on the [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) from Kaggle (918 patients, 11 clinical features).

### Models Implemented
| Model | F1 Score | AUC-ROC |
|-------|----------|---------|
| Logistic Regression | 0.9007 | 0.9344 |
| Decision Tree (CART) | 0.8701 | 0.8819 |
| Random Forest | 0.9020 | 0.9456 |
| **Gradient Boosting** | **0.9085** | **0.9393** |
| Neural Network (MLP) | 0.9007 | 0.9425 |

### Key Findings
- Gradient Boosting achieved the best F1 score (0.9085)
- ST_Slope, ChestPainType (Asymptomatic), and ExerciseAngina are the most predictive features
- SHAP analysis confirms model predictions align with clinical knowledge

## 🚀 How to Run

### 1. Clone and Install
```bash
git clone https://github.com/AveryJYL/msis522-hw1-heart-failure.git
cd msis522-hw1-heart-failure
pip install -r requirements.txt
```

### 2. Run Analysis (Optional)
```bash
python hw1_heart_failure.py
```

### 3. Run Streamlit App
```bash
streamlit run app.py
```

## 📁 Project Structure
```
├── app.py                          # Streamlit application
├── hw1_heart_failure.py            # Full analysis script
├── heart.csv                       # Dataset
├── requirements.txt                # Python dependencies
├── model_logistic_regression.joblib # Saved models
├── model_decision_tree.joblib
├── model_random_forest.joblib
├── model_gradient_boosting.joblib
├── model_mlp.joblib
├── scaler.joblib                   # StandardScaler
├── metadata.joblib                 # Best hyperparameters
├── feature_names.joblib            # Feature column names
├── model_comparison.csv            # Results summary table
└── fig_*.png                       # Generated visualizations
```

## 📝 Assignment
MSIS 522 - Advanced Analytics & Machine Learning  
Foster School of Business | University of Washington  
Instructor: Prof. Léonard Boussioux
