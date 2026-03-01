# ❤️ Heart Disease Prediction — MSIS 522 HW1

An end-to-end data science workflow for predicting heart disease using clinical features.

## 🔗 Live Demo

**[Streamlit App](https://msis522-hw1-heart-failure-gjfteetwerufsrd8tkvd63.streamlit.app/)**

## 📊 Overview

This project implements the complete data science pipeline on the [Heart Disease Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) from Kaggle (918 patients, 11 clinical features).

### Models & Results

| Model | F1 Score | AUC-ROC | Tuning |
|-------|----------|---------|--------|
| Logistic Regression (Baseline) | 0.9007 | 0.9344 | No CV (baseline) |
| Decision Tree (CART) | 0.8701 | 0.8819 | GridSearchCV 5-fold |
| **Random Forest** | **0.9055** | **0.9448** | GridSearchCV 5-fold |
| XGBoost | 0.8874 | 0.9388 | GridSearchCV 5-fold |
| Neural Network (Keras MLP) | 0.8616 | 0.9214 | 3-fold CV grid search (Bonus) |

> *Exact metrics are in `model_comparison.csv` and displayed in Tab 3 of the app.*

All tree-based models used `class_weight='balanced'` or `scale_pos_weight` to handle slight class imbalance. Hyperparameters were tuned via stratified cross-validation with `random_state=42`.

### Key Findings
- **ST_Slope**, **ExerciseAngina**, and **ChestPainType** are the most predictive features
- Asymptomatic patients have the highest heart disease rate (~79%)
- SHAP analysis confirms the model's predictions align with clinical cardiology knowledge

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
├── app.py                              # Streamlit app (4 tabs)
├── train.py                            # Complete training pipeline (EDA → train → save)
├── MSIS522_HW1_Heart_Failure.ipynb     # Jupyter notebook with full analysis
├── heart.csv                           # Dataset (918 rows, 12 columns)
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
│
├── model_logistic_regression.joblib    # Pre-trained: Logistic Regression
├── model_decision_tree.joblib          # Pre-trained: Decision Tree (CART)
├── model_random_forest.joblib          # Pre-trained: Random Forest
├── model_xgboost.joblib                # Pre-trained: XGBoost
├── model_mlp.keras                     # Pre-trained: Keras MLP
├── scaler.joblib                       # Fitted StandardScaler
├── metadata.joblib                     # Best hyperparameters for all models
├── model_comparison.csv                # Performance metrics table
│
├── fig_3_1_shap_summary.png            # SHAP beeswarm plot
├── fig_3_2_shap_bar.png                # SHAP feature importance bar
├── fig_3_3_shap_waterfall.png          # SHAP waterfall (high-risk example)
├── mlp_training_history.png            # Keras MLP loss & accuracy curves
└── mlp_tuning_results.png              # Bonus: MLP hyperparameter tuning chart
```

## 📝 Rubric Mapping

| Assignment Section | Where to Find It |
|---|---|
| **Part 1: Descriptive Analytics (25 pts)** | Notebook Part 1 + App Tab 2 |
| 1.1 Dataset Introduction | Notebook Section 1.1 + App Tab 1 |
| 1.2 Target Distribution | Notebook Section 1.2 + App Tab 2 |
| 1.3 Feature Visualizations (≥4) | Notebook Section 1.3 (6 visualizations) + App Tab 2 |
| 1.4 Correlation Heatmap | Notebook Section 1.4 + App Tab 2 |
| **Part 2: Predictive Analytics (45 pts)** | Notebook Part 2 + App Tab 3 |
| 2.2 Logistic Regression Baseline | Notebook Section 2.2 |
| 2.3 Decision Tree + GridSearchCV | Notebook Section 2.3 |
| 2.4 Random Forest + GridSearchCV | Notebook Section 2.4 |
| 2.5 XGBoost + GridSearchCV | Notebook Section 2.5 + `train.py` |
| 2.6 Keras MLP + Training History | Notebook Section 2.6 + `train.py` + `mlp_training_history.png` |
| 2.7 Model Comparison | Notebook Section 2.7 + App Tab 3 |
| **Part 3: SHAP (10 pts)** | Notebook Part 3 + App Tab 4 |
| **Part 4: Streamlit App (20 pts)** | App Tabs 1–4 |
| Tab 1 — Executive Summary | App Tab 1 |
| Tab 2 — Descriptive Analytics | App Tab 2 |
| Tab 3 — Model Performance & Hyperparams | App Tab 3 |
| Tab 4 — SHAP + Interactive Prediction + Waterfall | App Tab 4 |
| **Bonus: MLP Tuning (1 pt)** | `train.py` + `mlp_tuning_results.png` + App Tab 3 |

## 📚 Assignment
MSIS 522 — Advanced Analytics & Machine Learning  
Foster School of Business | University of Washington  
Instructor: Prof. Léonard Boussioux
