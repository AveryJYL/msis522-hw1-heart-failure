"""
MSIS 522 - HW1: Heart Disease Prediction - Streamlit App
Models: Logistic Regression, Decision Tree, Random Forest, XGBoost, Keras MLP
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
import joblib
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Heart Disease Prediction - MSIS 522 HW1", page_icon="❤️", layout="wide")

# ============================================================
# LOAD DATA AND MODELS
# ============================================================
@st.cache_data
def load_data():
    return pd.read_csv('heart.csv')

@st.cache_resource
def load_sklearn_models():
    models = {
        'Logistic Regression': joblib.load('model_logistic_regression.joblib'),
        'Decision Tree': joblib.load('model_decision_tree.joblib'),
        'Random Forest': joblib.load('model_random_forest.joblib'),
        'XGBoost': joblib.load('model_xgboost.joblib'),
    }
    scaler = joblib.load('scaler.joblib')
    metadata = joblib.load('metadata.joblib')
    return models, scaler, metadata

@st.cache_resource
def load_keras_model():
    """Try loading Keras model with graceful fallback."""
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model('model_mlp.h5')
        return model, None
    except Exception:
        pass
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model('model_mlp.keras')
        return model, None
    except Exception as e:
        return None, str(e)

df = load_data()
sklearn_models, scaler, metadata = load_sklearn_models()
keras_model, keras_error = load_keras_model()
results_df = pd.read_csv('model_comparison.csv')

# All models dict (exclude Keras if loading failed)
all_models = {**sklearn_models}
if keras_model is not None:
    all_models['Neural Network (Keras)'] = keras_model
else:
    st.info("ℹ️ The Keras MLP model was trained offline in Colab (see Notebook Section 2.6). "
            "Its full metrics are shown in Tab 3. The deployment environment uses the 4 compatible models for live inference.")

# Prepare test set (same split as training)
df_model = pd.get_dummies(df, columns=['Sex', 'ChestPainType', 'RestingECG',
                                        'ExerciseAngina', 'ST_Slope'], drop_first=True)
X = df_model.drop('HeartDisease', axis=1)
y = df_model['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scale using ONLY transform (scaler was fit during training, not here)
num_cols = metadata['num_cols']
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[num_cols] = scaler.transform(X_train[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

# Models that need scaled input
SCALED_MODELS = {'Logistic Regression', 'Neural Network (Keras)'}

# Determine best model
best_model_name = results_df.loc[results_df['F1'].idxmax(), 'Model']
best_f1 = results_df['F1'].max()
best_auc = results_df.loc[results_df['F1'].idxmax(), 'AUC-ROC']

# Tree-based models (for SHAP waterfall)
TREE_MODELS = {'Decision Tree', 'Random Forest', 'XGBoost'}

# ============================================================
# HELPER: get predictions from any model
# ============================================================
def get_predictions(model_name, X_input):
    """Return (predicted_class, probability_array) for any model."""
    model = all_models[model_name]
    if model_name in SCALED_MODELS:
        inp = X_input.copy()
        inp[num_cols] = scaler.transform(X_input[num_cols])
    else:
        inp = X_input

    if model_name == 'Neural Network (Keras)':
        prob_1 = model.predict(inp.values, verbose=0).flatten()[0]
        pred = int(prob_1 >= 0.5)
        prob = np.array([1 - prob_1, prob_1])
    else:
        pred = model.predict(inp)[0]
        prob = model.predict_proba(inp)[0]
    return pred, prob

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
    st.title("❤️ Heart Disease Prediction")
    st.markdown("### An End-to-End Machine Learning Analysis")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Size", "918 patients")
    c2.metric("Features", "11 clinical")
    c3.metric("Best Model F1", f"{best_f1:.1%}")
    c4.metric("Best AUC-ROC", f"{best_auc:.1%}")
    st.divider()

    st.markdown(f"""
    ### The Problem
    Cardiovascular diseases (CVDs) are the **#1 cause of death globally**, taking an estimated
    17.9 million lives each year. Early detection and risk stratification are critical for
    improving patient outcomes.

    ### The Dataset
    The **Heart Disease Prediction Dataset** from Kaggle combines five heart disease databases
    (Cleveland, Hungarian, Switzerland, Long Beach VA, Statlog) with **918 patient records**
    and 11 clinical features including age, blood pressure, cholesterol, and exercise test results.

    ### Our Approach
    We built and compared **five ML models** — Logistic Regression, Decision Tree, Random Forest,
    XGBoost, and a Keras Neural Network — each tuned using 5-fold stratified cross-validation.
    Tree-based models used `class_weight='balanced'` or `scale_pos_weight` to handle the slight
    class imbalance.

    ### Key Findings
    - **{best_model_name}** achieved the best performance (F1: **{best_f1:.1%}**, AUC-ROC: **{best_auc:.1%}**)
    - Most predictive features: **ST_Slope**, **ExerciseAngina**, **ChestPainType**
    - **Asymptomatic patients** have the highest heart disease rate (~79%)
    - SHAP analysis confirms the model aligns with established clinical knowledge

    ### So What?
    This model could help clinicians **prioritize high-risk patients** for further cardiac testing.
    The interactive tool (Tab 4) demonstrates clinical decision support in practice — input patient
    data and get a real-time risk prediction with SHAP explanation.
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

    st.info("Slightly imbalanced (55.3% vs 44.7%). Using `class_weight='balanced'` and F1/AUC-ROC as primary metrics.")
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
    st.caption("**Insight:** ASY (Asymptomatic) → ~79% heart disease. No symptoms ≠ no risk.")

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
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax)
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

    # Best Hyperparameters
    st.markdown("### Best Hyperparameters (from 5-Fold GridSearchCV)")
    hp_col1, hp_col2 = st.columns(2)
    with hp_col1:
        st.markdown("**Logistic Regression (Baseline):**")
        st.json(metadata['best_params']['Logistic Regression'])
        st.markdown("**Decision Tree:**")
        st.json(metadata['best_params']['Decision Tree'])
        st.markdown("**Random Forest:**")
        st.json(metadata['best_params']['Random Forest'])
    with hp_col2:
        st.markdown("**XGBoost:**")
        st.json(metadata['best_params']['XGBoost'])
        st.markdown("**Neural Network (Keras MLP):**")
        st.json(metadata['best_params']['MLP'])

    st.divider()

    # Performance Bar Chart
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

    # ROC Curves
    st.markdown("### ROC Curves")
    fig, ax = plt.subplots(figsize=(10, 8))
    for name in all_models:
        X_eval = X_test_scaled if name in SCALED_MODELS else X_test
        model = all_models[name]
        if name == 'Neural Network (Keras)':
            y_prob = model.predict(X_eval.values, verbose=0).flatten()
        else:
            y_prob = model.predict_proba(X_eval)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc(fpr, tpr):.3f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - All Models', fontsize=14, fontweight='bold'); ax.legend()
    st.pyplot(fig); plt.close()

    st.divider()

    # Confusion Matrices
    st.markdown("### Confusion Matrices")
    fig, axes = plt.subplots(1, 5, figsize=(25, 4))
    for ax, name in zip(axes, all_models):
        X_eval = X_test_scaled if name in SCALED_MODELS else X_test
        model = all_models[name]
        if name == 'Neural Network (Keras)':
            y_pred = (model.predict(X_eval.values, verbose=0).flatten() >= 0.5).astype(int)
        else:
            y_pred = model.predict(X_eval)
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=['Normal', 'HD']).plot(ax=ax, cmap='Blues')
        ax.set_title(name, fontsize=9, fontweight='bold')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.divider()

    # MLP Training History
    st.markdown("### MLP Training History")
    try:
        st.image('mlp_training_history.png', use_container_width=True)
        st.caption("Left: Binary cross-entropy loss. Right: Accuracy. Early stopping prevents overfitting.")
    except:
        st.info("Training history image not available.")

    # MLP Tuning Results (Bonus)
    st.markdown("### Bonus: MLP Hyperparameter Tuning")
    try:
        st.image('mlp_tuning_results.png', use_container_width=True)
        st.caption("Grid search over hidden layer sizes, dropout rates, and learning rates. Best config in red.")
    except:
        st.info("Tuning results image not available.")

    st.divider()

    st.markdown("### Analysis")
    st.markdown(f"""
    **{best_model_name}** achieved the best F1 ({best_f1:.4f}) and AUC-ROC ({best_auc:.4f}).
    Ensemble and boosting methods outperformed single models because they combine many weak learners.
    **Logistic Regression** performed surprisingly well as a simple baseline, suggesting the features
    have a fairly linear relationship with heart disease risk. **Decision Tree** was weakest — a
    single tree memorizes training data and doesn't generalize as well. **XGBoost** was competitive,
    benefiting from sequential boosting and regularization. The **Keras MLP** improved after
    hyperparameter tuning over hidden layer sizes, dropout rates, and learning rates.
    """)

# ============================================================
# TAB 4: EXPLAINABILITY & INTERACTIVE PREDICTION
# ============================================================
with tab4:
    st.title("🔍 Explainability & Interactive Prediction")

    st.markdown("### SHAP Analysis (Best Tree Model)")

    st.markdown("#### SHAP Summary Plot (Beeswarm)")
    st.image('fig_3_1_shap_summary.png', use_container_width=True)
    st.caption("Red = high feature value, Blue = low. Points right of center → increase heart disease probability.")

    st.markdown("#### SHAP Feature Importance")
    st.image('fig_3_2_shap_bar.png', use_container_width=True)

    st.markdown("#### SHAP Waterfall — High Risk Patient Example")
    st.image('fig_3_3_shap_waterfall.png', use_container_width=True)
    st.caption("Shows how each feature pushed the prediction from the base rate to a 99.5% risk for one patient.")

    st.divider()
    st.markdown("### SHAP Interpretation")
    st.markdown("""
    **Top features:** ST_Slope (Up/Flat), ExerciseAngina, MaxHR, Cholesterol, Oldpeak — all related
    to how the heart performs during exercise, which aligns with clinical cardiology practice.

    **Direction:** Flat ST_Slope + exercise angina + high Oldpeak → **increases** heart disease risk.
    Upward ST_Slope + high MaxHR → **decreases** risk (signs of a healthy heart).

    **Clinical value:** The model's most important features match what cardiologists actually look at.
    This tool could help flag high-risk patients — especially asymptomatic ones — for further testing.
    """)

    st.divider()

    # Interactive Prediction
    st.markdown("### 🎯 Interactive Prediction Tool")
    st.markdown("Adjust the inputs below to get a real-time heart disease risk prediction with SHAP explanation.")

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
            format_func=lambda x: {"ASY": "ASY — Asymptomatic", "ATA": "ATA — Atypical Angina",
                                    "NAP": "NAP — Non-Anginal", "TA": "TA — Typical Angina"}[x])
    with c3:
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "LVH", "ST"])
        exercise_angina = st.selectbox("Exercise Angina", ["N", "Y"], format_func=lambda x: "Yes" if x == "Y" else "No")
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
        selected_model = st.selectbox("Model", list(all_models.keys()))

    # Build input DataFrame
    input_data = pd.DataFrame({
        'Age': [age], 'RestingBP': [resting_bp], 'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs], 'MaxHR': [max_hr], 'Oldpeak': [oldpeak],
        'Sex_M': [1 if sex == 'M' else 0],
        'ChestPainType_ATA': [1 if chest_pain == 'ATA' else 0],
        'ChestPainType_NAP': [1 if chest_pain == 'NAP' else 0],
        'ChestPainType_TA': [1 if chest_pain == 'TA' else 0],
        'RestingECG_Normal': [1 if resting_ecg == 'Normal' else 0],
        'RestingECG_ST': [1 if resting_ecg == 'ST' else 0],
        'ExerciseAngina_Y': [1 if exercise_angina == 'Y' else 0],
        'ST_Slope_Flat': [1 if st_slope == 'Flat' else 0],
        'ST_Slope_Up': [1 if st_slope == 'Up' else 0],
    })

    pred, prob = get_predictions(selected_model, input_data)

    st.divider()
    st.markdown("### Prediction Result")
    r1, r2, r3 = st.columns(3)
    with r1:
        if pred == 1:
            st.error("⚠️ **Heart Disease Detected**")
        else:
            st.success("✅ **No Heart Disease**")
    with r2:
        st.metric("Risk Probability", f"{prob[1]:.1%}")
    with r3:
        st.metric("Model Used", selected_model)

    # Risk bar
    fig, ax = plt.subplots(figsize=(8, 1.5))
    ax.barh([0], [prob[1]], color='#e74c3c' if prob[1] > 0.5 else '#2ecc71', height=0.5)
    ax.barh([0], [1 - prob[1]], left=[prob[1]], color='#ecf0f1', height=0.5)
    ax.set_xlim(0, 1); ax.set_yticks([]); ax.set_xlabel('Heart Disease Probability')
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1)
    st.pyplot(fig); plt.close()

    # SHAP Waterfall — linked to model selection
    st.markdown("### SHAP Waterfall for Your Custom Input")

    if selected_model in TREE_MODELS:
        try:
            import shap
            tree_model = all_models[selected_model]
            explainer = shap.TreeExplainer(tree_model)
            input_float = input_data.astype(float)
            shap_vals = explainer.shap_values(input_float, check_additivity=False)

            if isinstance(shap_vals, list):
                sv = shap_vals[1][0]
                base = explainer.expected_value[1]
            elif shap_vals.ndim == 3:
                sv = shap_vals[0, :, 1]
                base = explainer.expected_value[1] if hasattr(explainer.expected_value, '__len__') else explainer.expected_value
            else:
                sv = shap_vals[0]
                base = explainer.expected_value if not hasattr(explainer.expected_value, '__len__') else explainer.expected_value[1]

            fig, ax = plt.subplots(figsize=(12, 8))
            shap.waterfall_plot(
                shap.Explanation(
                    values=sv,
                    base_values=float(base),
                    data=input_float.iloc[0].values,
                    feature_names=input_data.columns.tolist()
                ),
                show=False
            )
            plt.title(f'SHAP Waterfall — {selected_model} — Your Custom Input', fontsize=12, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig); plt.close()
            st.caption("Red bars push toward Heart Disease, blue bars push toward Normal. "
                       "This explains exactly why the model made this prediction for your input.")
        except ImportError:
            st.warning("⚠️ SHAP library not installed. Install with `pip install shap` to see waterfall plots.")
        except Exception as e:
            st.warning(f"Could not generate waterfall: {e}")
    else:
        st.info(f"💡 SHAP waterfall is available for tree-based models (Decision Tree, Random Forest, XGBoost). "
                f"Currently showing prediction from {selected_model}. "
                f"The static SHAP plots above use the best tree model for interpretability.")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown(
    "<div style='text-align:center;color:gray;font-size:0.8em;'>"
    "MSIS 522 HW1 | Heart Disease Prediction | "
    "<a href='https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction'>Kaggle Dataset</a> | "
    "<a href='https://github.com/AveryJYL/msis522-hw1-heart-failure'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True
)
