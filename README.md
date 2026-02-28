# ❤️ Heart Disease Prediction - MSIS 522 HW1

An end-to-end data science workflow for predicting heart disease using clinical features.

## 🔗 Live Demo

[**Streamlit App**](https://msis522-hw1-heart-failure-gjfteetwerufsrd8tkvd63.streamlit.app/)

## 📊 Overview

This project implements the complete data science pipeline on the [Heart Disease Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) from Kaggle (918 patients, 11 clinical features).

### Models Implemented
| Model | F1 Score | AUC-ROC |
|-------|----------|---------|
| Logistic Regression | 0.9007 | 0.9344 |
| Decision Tree (CART) | 0.8701 | 0.8819 |
| **Random Forest** | **0.9055** | **0.9448** |
| Gradient Boosting | 0.8875 | 0.9397 |
| Neural Network (MLP) | 0.8889 | 0.9382 |

All tree-based models used `class_weight='balanced'` to handle slight class imbalance. Hyperparameters were tuned via 5-fold Stratified GridSearchCV with `random_state=42`.

### Key Findings
- Random Forest achieved the best F1 score (0.9055) and AUC-ROC (0.9448)
- ST_Slope, ChestPainType (Asymptomatic), and ExerciseAngina are the most predictive features
- SHAP analysis confirms model predictions align with clinical knowledge

## 🚀 How to Run

### 1. Clone and Install
```bash
git clone https://github.com/AveryJYL/msis522-hw1-heart-failure.git
cd msis522-hw1-heart-failure
pip install -r requirements.txt
```

### 2. Run Streamlit App
```bash
streamlit run app.py
```

## 📁 Project Structure
```
├── app.py                              # Streamlit application (4 tabs)
├── heart.csv                           # Dataset (918 rows, 12 columns)
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── model_comparison.csv                # Model performance summary table
├── model_logistic_regression.joblib    # Saved models (pre-trained)
├── model_decision_tree.joblib
├── model_random_forest.joblib
├── model_gradient_boosting.joblib
├── model_mlp.joblib
├── scaler.joblib                       # StandardScaler for numerical features
├── metadata.joblib                     # Best hyperparameters for all models
├── fig_3_1_shap_summary.png            # SHAP beeswarm plot
├── fig_3_2_shap_bar.png                # SHAP feature importance bar plot
└── fig_3_3_shap_waterfall.png          # SHAP waterfall (high-risk patient example)
```

## 📝 Assignment
MSIS 522 - Advanced Analytics & Machine Learning  
Foster School of Business | University of Washington  
Instructor: Prof. Léonard Boussioux
