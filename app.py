"""
MSIS 522 - HW1: Heart Failure Prediction - Streamlit App
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay)
from PIL import Image
import joblib
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Heart Failure Prediction - MSIS 522 HW1", page_icon="❤️", layout="wide")

# ============================================================
# LOAD DATA AND MODELS
# ============================================================
@st.cache_data
def load_data():
    return pd.read_csv('heart.csv')

@st.cache_resource
def load_models():
    models = {
        'Logistic Regression': joblib.load('model_logistic_regression.joblib'),
        'Decision Tree': joblib.load('model_decision_tree.joblib'),
        'Random Forest': joblib.load('model_random_forest.joblib'),
        'Gradient Boosting': joblib.load('model_gradient_boosting.joblib'),
        'Neural Network (MLP)': joblib.load('model_mlp.joblib'),
    }
    scaler = joblib.load('scaler.joblib')
    return models, scaler

df = load_data()
models, scaler = load_models()
results_df = pd.read_csv('model_comparison.csv')

# Prepare test set (same split as training)
df_model = pd.get_dummies(df, columns=['Sex', 'ChestPainType', 'RestingECG',
                                        'ExerciseAngina', 'ST_Slope'], drop_first=True)
X = df_model.drop('HeartDisease', axis=1)
y = df_model['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
num_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
X_test_scaled = X_test.copy()
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Executive Summary", "📊 Descriptive Analytics",
    "🤖 Model Performance", "🔍 Explainability & Prediction"
])

# ============================================================
# TAB 1: EXECUTIVE SUMMARY
# ============================================================
with tab1:
    st.title("❤️ Heart Failure Prediction")
    st.markdown("### An End-to-End Machine Learning Analysis")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Size", "918 patients")
    c2.metric("Features", "11 clinical")
    c3.metric("Best Model F1", f"{results_df['F1'].max():.1%}")
    c4.metric("Best AUC-ROC", f"{results_df['AUC-ROC'].max():.1%}")
    st.divider()

    st.markdown(f"""
    ### The Problem
    Cardiovascular diseases (CVDs) are the **#1 cause of death globally**, taking an estimated
    17.9 million lives each year. Early detection and risk stratification are critical for
    improving patient outcomes.

    ### The Dataset
    The **Heart Failure Prediction Dataset** from Kaggle combines five heart disease databases
    (Cleveland, Hungarian, Switzerland, Long Beach VA, Statlog) with **918 patient records**
    and 11 clinical features.

    ### Our Approach
    We built and compared **five ML models** — from Logistic Regression to Gradient Boosted
    Trees and Neural Networks — each tuned using 5-fold stratified cross-validation.

    ### Key Findings
    - **Random Forest** achieved the best performance (F1: **{results_df['F1'].max():.1%}**, AUC-ROC: **{results_df['AUC-ROC'].max():.1%}**)
    - Most predictive features: **ST_Slope**, **ExerciseAngina**, **ChestPainType**
    - **Asymptomatic patients** have the highest heart disease rate (~79%)
    - SHAP analysis confirms model aligns with clinical knowledge

    ### So What?
    This model could help clinicians **prioritize high-risk patients** for further testing.
    The interactive tool (Tab 4) demonstrates clinical decision support in practice.
    """)

# ============================================================
# TAB 2: DESCRIPTIVE ANALYTICS
# ============================================================
with tab2:
    st.title("📊 Descriptive Analytics")

    st.markdown("### Target Variable Distribution")
    c1, c2 = st.columns(2)
    target_counts = df['HeartDisease'].value_counts()
    colors = ['#2ecc71', '#e74c3c']

    with c1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(['Normal (0)', 'Heart Disease (1)'], target_counts.values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title('Heart Disease Distribution', fontsize=14, fontweight='bold')
        ax.set_ylabel('Count')
        for i, v in enumerate(target_counts.values):
            ax.text(i, v + 10, f'{v} ({v/len(df)*100:.1f}%)', ha='center', fontweight='bold')
        st.pyplot(fig); plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.pie(target_counts.values, labels=['Normal', 'Heart Disease'], colors=colors,
               autopct='%1.1f%%', startangle=90, explode=(0.05, 0.05), textprops={'fontsize': 12})
        ax.set_title('Target Proportion', fontsize=14, fontweight='bold')
        st.pyplot(fig); plt.close()

    st.info("Slightly imbalanced (55.3% vs 44.7%). Using `class_weight='balanced'` and F1/AUC-ROC metrics.")
    st.divider()

    st.markdown("### Feature Distributions & Relationships")

    st.markdown("#### 1. Age Distribution")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=df, x='Age', hue='HeartDisease', kde=True, bins=30, palette=['#2ecc71', '#e74c3c'], alpha=0.6, ax=ax)
    ax.set_title('Age Distribution by Heart Disease', fontsize=14, fontweight='bold')
    ax.legend(title='Heart Disease', labels=['Normal', 'Heart Disease'])
    st.pyplot(fig); plt.close()
    st.caption("**Insight:** Heart disease patients peak at 55-65. Normal patients skew younger.")

    st.markdown("#### 2. Chest Pain Type")
    fig, ax = plt.subplots(figsize=(10, 5))
    ct = pd.crosstab(df['ChestPainType'], df['HeartDisease'], normalize='index') * 100
    ct.plot(kind='bar', stacked=True, color=['#2ecc71', '#e74c3c'], ax=ax, edgecolor='black', linewidth=0.5)
    ax.set_title('Heart Disease Rate by Chest Pain Type', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage (%)')
    ax.legend(title='Heart Disease', labels=['Normal', 'Heart Disease'])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    for c in ax.containers: ax.bar_label(c, fmt='%.1f%%', label_type='center', fontsize=9)
    st.pyplot(fig); plt.close()
    st.caption("**Insight:** ASY (Asymptomatic) → ~79% heart disease. Symptom-free patients are most at risk.")

    st.markdown("#### 3. Numerical Features")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, feat, title in zip(axes.flatten(), ['MaxHR', 'Oldpeak', 'RestingBP', 'Cholesterol'],
                                ['Max Heart Rate', 'Oldpeak', 'Resting BP', 'Cholesterol']):
        sns.boxplot(data=df, x='HeartDisease', y=feat, hue='HeartDisease',
                    palette=['#2ecc71', '#e74c3c'], ax=ax, legend=False)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticklabels(['Normal', 'Heart Disease']); ax.set_xlabel('')
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.caption("**Insight:** Heart disease → lower MaxHR, higher Oldpeak. Cholesterol has zero-value anomalies.")

    st.markdown("#### 4. Exercise Angina & ST Slope")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, title in zip(axes, ['ExerciseAngina', 'ST_Slope'], ['Exercise Angina', 'ST Slope']):
        ct_t = pd.crosstab(df[col], df['HeartDisease'], normalize='index') * 100
        ct_t.plot(kind='bar', stacked=True, color=['#2ecc71', '#e74c3c'], ax=ax, edgecolor='black', linewidth=0.5)
        ax.set_title(f'Heart Disease Rate by {title}', fontsize=13, fontweight='bold')
        ax.set_ylabel('Percentage (%)'); ax.legend(title='HD', labels=['Normal', 'HD'])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.caption("**Insight:** Exercise Angina (Yes) → ~77%. Flat ST_Slope → ~84%. Strongest indicators.")

    st.divider()
    st.markdown("### Correlation Heatmap")
    df_enc = df.copy()
    for col in ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']:
        df_enc[col] = LabelEncoder().fit_transform(df_enc[col])
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df_enc.corr(); mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax)
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    st.pyplot(fig); plt.close()
    st.caption("**Insight:** ST_Slope (0.52), ExerciseAngina (0.49), Oldpeak (0.40), MaxHR (-0.40) strongest with HeartDisease.")

# ============================================================
# TAB 3: MODEL PERFORMANCE
# ============================================================
with tab3:
    st.title("🤖 Model Performance")

    st.markdown("### Model Comparison")
    st.dataframe(results_df.style.highlight_max(
        subset=['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC'], color='#90EE90'
    ).format({c: '{:.4f}' for c in ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC']}),
        use_container_width=True)
    st.divider()

    st.markdown("### Performance Bar Chart")
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(results_df)); w = 0.15
    for i, (m, c) in enumerate(zip(['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC'],
                                     ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'])):
        ax.bar(x + i*w, results_df[m], w, label=m, color=c, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Score'); ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + w*2); ax.set_xticklabels(results_df['Model'], fontsize=9)
    ax.legend(); ax.set_ylim(0.7, 1.0)
    st.pyplot(fig); plt.close()
    st.divider()

    st.markdown("### ROC Curves")
    fig, ax = plt.subplots(figsize=(10, 8))
    for name, model in models.items():
        X_eval = X_test_scaled if name in ['Logistic Regression', 'Neural Network (MLP)'] else X_test
        y_prob = model.predict_proba(X_eval)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc(fpr, tpr):.3f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - All Models', fontsize=14, fontweight='bold'); ax.legend()
    st.pyplot(fig); plt.close()
    st.divider()

    st.markdown("### Confusion Matrices")
    fig, axes = plt.subplots(1, 5, figsize=(25, 4))
    for ax, (name, model) in zip(axes, models.items()):
        X_eval = X_test_scaled if name in ['Logistic Regression', 'Neural Network (MLP)'] else X_test
        cm = confusion_matrix(y_test, model.predict(X_eval))
        ConfusionMatrixDisplay(cm, display_labels=['Normal', 'HD']).plot(ax=ax, cmap='Blues')
        ax.set_title(name, fontsize=9, fontweight='bold')
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.divider()

    st.markdown("### Analysis")
    st.markdown("""
    **Random Forest** achieved the best F1 (0.9055) and AUC-ROC (0.9448). **Logistic Regression**
    performed surprisingly well as a baseline. **Decision Tree** was weakest — a single tree
    can't capture complex interactions like ensembles. Gradient Boosting and MLP performed well
    but didn't surpass Random Forest here.
    """)

# ============================================================
# TAB 4: EXPLAINABILITY & INTERACTIVE PREDICTION
# ============================================================
with tab4:
    st.title("🔍 Explainability & Interactive Prediction")

    st.markdown("### SHAP Analysis (Random Forest)")

    st.markdown("#### SHAP Summary Plot (Beeswarm)")
    st.image('fig_3_1_shap_summary.png', use_container_width=True)
    st.caption("Red = high feature value, Blue = low. Points right of center → increase heart disease probability.")

    st.markdown("#### SHAP Feature Importance")
    st.image('fig_3_2_shap_bar.png', use_container_width=True)

    st.markdown("#### SHAP Waterfall - High Risk Patient")
    st.image('fig_3_3_shap_waterfall.png', use_container_width=True)
    st.caption("Shows how each feature pushed prediction from base (0.501) to final (0.995) for one high-risk patient.")

    st.divider()
    st.markdown("### SHAP Interpretation")
    st.markdown("""
    **Top features:** ST_Slope (Up/Flat), ExerciseAngina, MaxHR, Cholesterol, Oldpeak.
    
    **Direction:** Flat ST_Slope, exercise angina, high Oldpeak → **increases** risk.
    Up ST_Slope, high MaxHR → **decreases** risk.
    
    **Clinical value:** Aligns with cardiology knowledge. Patients with flat ST slope +
    exercise angina should be flagged for further testing.
    """)

    st.divider()
    st.markdown("### 🎯 Interactive Prediction Tool")

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age", 28, 77, 54)
        resting_bp = st.slider("Resting BP (mmHg)", 80, 200, 130)
        cholesterol = st.slider("Cholesterol (mg/dl)", 0, 603, 200)
        fasting_bs = st.selectbox("Fasting BS > 120", [0, 1], format_func=lambda x: "Yes" if x else "No")
    with c2:
        max_hr = st.slider("Max Heart Rate", 60, 202, 140)
        oldpeak = st.slider("Oldpeak", -2.6, 6.2, 0.0, step=0.1)
        sex = st.selectbox("Sex", ["M", "F"])
        chest_pain = st.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"],
            format_func=lambda x: {"ASY":"ASY-Asymptomatic","ATA":"ATA-Atypical Angina",
                                    "NAP":"NAP-Non-Anginal","TA":"TA-Typical Angina"}[x])
    with c3:
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "LVH", "ST"])
        exercise_angina = st.selectbox("Exercise Angina", ["N", "Y"], format_func=lambda x: "Yes" if x=="Y" else "No")
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
        selected_model = st.selectbox("Model", list(models.keys()))

    input_data = pd.DataFrame({
        'Age': [age], 'RestingBP': [resting_bp], 'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs], 'MaxHR': [max_hr], 'Oldpeak': [oldpeak],
        'Sex_M': [1 if sex=='M' else 0],
        'ChestPainType_ATA': [1 if chest_pain=='ATA' else 0],
        'ChestPainType_NAP': [1 if chest_pain=='NAP' else 0],
        'ChestPainType_TA': [1 if chest_pain=='TA' else 0],
        'RestingECG_Normal': [1 if resting_ecg=='Normal' else 0],
        'RestingECG_ST': [1 if resting_ecg=='ST' else 0],
        'ExerciseAngina_Y': [1 if exercise_angina=='Y' else 0],
        'ST_Slope_Flat': [1 if st_slope=='Flat' else 0],
        'ST_Slope_Up': [1 if st_slope=='Up' else 0],
    })

    model = models[selected_model]
    if selected_model in ['Logistic Regression', 'Neural Network (MLP)']:
        inp = input_data.copy(); inp[num_cols] = scaler.transform(input_data[num_cols])
        pred = model.predict(inp)[0]; prob = model.predict_proba(inp)[0]
    else:
        pred = model.predict(input_data)[0]; prob = model.predict_proba(input_data)[0]

    st.divider()
    st.markdown("### Prediction Result")
    r1, r2, r3 = st.columns(3)
    with r1:
        if pred == 1: st.error("⚠️ **Heart Disease Detected**")
        else: st.success("✅ **No Heart Disease**")
    with r2: st.metric("Risk Probability", f"{prob[1]:.1%}")
    with r3: st.metric("Model", selected_model)

    fig, ax = plt.subplots(figsize=(8, 1.5))
    ax.barh([0], [prob[1]], color='#e74c3c' if prob[1]>0.5 else '#2ecc71', height=0.5)
    ax.barh([0], [1-prob[1]], left=[prob[1]], color='#ecf0f1', height=0.5)
    ax.set_xlim(0,1); ax.set_yticks([]); ax.set_xlabel('Heart Disease Probability')
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1)
    st.pyplot(fig); plt.close()

st.divider()
st.markdown("<div style='text-align:center;color:gray;font-size:0.8em;'>MSIS 522 HW1 | Heart Failure Prediction | "
            "<a href='https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction'>Kaggle Dataset</a></div>",
            unsafe_allow_html=True)
