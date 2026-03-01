"""
MSIS 522 HW1 - Complete Training Pipeline (Fixed)
Run in Google Colab: !python train_v2.py
Make sure heart.csv is in the same directory.
"""

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "xgboost", "shap"])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import joblib
import random
import warnings
warnings.filterwarnings('ignore')

# Reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow: {tf.__version__}")
print(f"XGBoost: {xgb.__version__}")

# ============================================================
# 1. LOAD & PREPARE DATA
# ============================================================
df = pd.read_csv('heart.csv')
print(f"Dataset: {df.shape[0]} rows x {df.shape[1]} columns")

df_model = pd.get_dummies(df, columns=['Sex', 'ChestPainType', 'RestingECG',
                                        'ExerciseAngina', 'ST_Slope'], drop_first=True)

# FIX: Convert bool columns to int immediately
for col in df_model.columns:
    if df_model[col].dtype == 'bool':
        df_model[col] = df_model[col].astype(int)

X = df_model.drop('HeartDisease', axis=1)
y = df_model['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
num_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
X_train_scaled = X_train.copy().astype(float)
X_test_scaled = X_test.copy().astype(float)
X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

joblib.dump(scaler, 'scaler.joblib')
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print(f"Dtypes: {X_train_scaled.dtypes.unique()}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_results = []
best_params_dict = {}

def evaluate(model, X_eval, y_eval, name):
    y_pred = model.predict(X_eval)
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_eval)[:, 1]
    else:
        y_prob = model.predict(X_eval)
    m = {
        'Model': name,
        'Accuracy': accuracy_score(y_eval, y_pred),
        'Precision': precision_score(y_eval, y_pred),
        'Recall': recall_score(y_eval, y_pred),
        'F1': f1_score(y_eval, y_pred),
        'AUC-ROC': roc_auc_score(y_eval, y_prob)
    }
    print(f"\n{'='*50}\n  {name}\n{'='*50}")
    for k, v in m.items():
        if k != 'Model': print(f"  {k}: {v:.4f}")
    return m

# ============================================================
# 2.2 LOGISTIC REGRESSION
# ============================================================
lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
lr.fit(X_train_scaled, y_train)
all_results.append(evaluate(lr, X_test_scaled, y_test, 'Logistic Regression'))
joblib.dump(lr, 'model_logistic_regression.joblib')
best_params_dict['Logistic Regression'] = {'max_iter': 1000, 'class_weight': 'balanced'}

# ============================================================
# 2.3 DECISION TREE
# ============================================================
dt_gs = GridSearchCV(
    DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    {'max_depth': [3, 5, 7, 10], 'min_samples_leaf': [5, 10, 20, 50]},
    cv=cv, scoring='f1', n_jobs=-1
)
dt_gs.fit(X_train, y_train)
print(f"\nDT Best: {dt_gs.best_params_}, CV F1: {dt_gs.best_score_:.4f}")
all_results.append(evaluate(dt_gs.best_estimator_, X_test, y_test, 'Decision Tree'))
joblib.dump(dt_gs.best_estimator_, 'model_decision_tree.joblib')
best_params_dict['Decision Tree'] = dt_gs.best_params_

# ============================================================
# 2.4 RANDOM FOREST
# ============================================================
rf_gs = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 8]},
    cv=cv, scoring='f1', n_jobs=-1
)
rf_gs.fit(X_train, y_train)
print(f"\nRF Best: {rf_gs.best_params_}, CV F1: {rf_gs.best_score_:.4f}")
all_results.append(evaluate(rf_gs.best_estimator_, X_test, y_test, 'Random Forest'))
joblib.dump(rf_gs.best_estimator_, 'model_random_forest.joblib')
best_params_dict['Random Forest'] = rf_gs.best_params_

# ============================================================
# 2.5 XGBOOST
# ============================================================
scale_pos = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
xgb_gs = GridSearchCV(
    xgb.XGBClassifier(random_state=42, scale_pos_weight=scale_pos,
                       eval_metric='logloss', use_label_encoder=False),
    {'n_estimators': [50, 100, 200], 'max_depth': [3, 4, 5, 6], 'learning_rate': [0.01, 0.05, 0.1]},
    cv=cv, scoring='f1', n_jobs=-1
)
xgb_gs.fit(X_train, y_train)
print(f"\nXGB Best: {xgb_gs.best_params_}, CV F1: {xgb_gs.best_score_:.4f}")
all_results.append(evaluate(xgb_gs.best_estimator_, X_test, y_test, 'XGBoost'))
joblib.dump(xgb_gs.best_estimator_, 'model_xgboost.joblib')
best_params_dict['XGBoost'] = xgb_gs.best_params_

# ============================================================
# 2.6 KERAS MLP
# ============================================================
print(f"\n{'='*60}\n  2.6 NEURAL NETWORK (KERAS)\n{'='*60}")

def build_keras_model(h1=128, h2=128, drop=0.3, lr=0.001):
    model = keras.Sequential([
        layers.Input(shape=(X_train_scaled.shape[1],)),
        layers.Dense(h1, activation='relu'),
        layers.Dropout(drop),
        layers.Dense(h2, activation='relu'),
        layers.Dropout(drop),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

keras_model = build_keras_model(128, 128, 0.3, 0.001)
history = keras_model.fit(
    X_train_scaled.values, y_train.values.astype(float),
    epochs=100, batch_size=32, validation_split=0.15,
    callbacks=[callbacks.EarlyStopping(patience=15, restore_best_weights=True),
               callbacks.ReduceLROnPlateau(patience=5, factor=0.5)],
    verbose=1
)

# Training history plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2, linestyle='--')
axes[0].set_title('Loss', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].legend()
axes[1].plot(history.history['accuracy'], label='Train Acc', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Val Acc', linewidth=2, linestyle='--')
axes[1].set_title('Accuracy', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy'); axes[1].legend()
plt.tight_layout()
plt.savefig('mlp_training_history.png', dpi=150, bbox_inches='tight')
plt.show()

y_prob_base = keras_model.predict(X_test_scaled.values).flatten()
y_pred_base = (y_prob_base >= 0.5).astype(int)
keras_base = {
    'Model': 'Neural Network (Keras)',
    'Accuracy': accuracy_score(y_test, y_pred_base),
    'Precision': precision_score(y_test, y_pred_base),
    'Recall': recall_score(y_test, y_pred_base),
    'F1': f1_score(y_test, y_pred_base),
    'AUC-ROC': roc_auc_score(y_test, y_prob_base)
}
print(f"\nBase MLP F1: {keras_base['F1']:.4f}")

# ============================================================
# BONUS: KERAS TUNING
# ============================================================
print(f"\n{'='*60}\n  BONUS: KERAS HYPERPARAMETER TUNING\n{'='*60}")

configs = [
    {'h1': 64,  'h2': 64,  'drop': 0.2, 'lr': 0.001},
    {'h1': 64,  'h2': 64,  'drop': 0.3, 'lr': 0.001},
    {'h1': 128, 'h2': 64,  'drop': 0.2, 'lr': 0.001},
    {'h1': 128, 'h2': 64,  'drop': 0.3, 'lr': 0.001},
    {'h1': 128, 'h2': 128, 'drop': 0.2, 'lr': 0.001},
    {'h1': 128, 'h2': 128, 'drop': 0.3, 'lr': 0.001},
    {'h1': 64,  'h2': 64,  'drop': 0.2, 'lr': 0.01},
    {'h1': 64,  'h2': 64,  'drop': 0.3, 'lr': 0.01},
    {'h1': 128, 'h2': 64,  'drop': 0.2, 'lr': 0.01},
    {'h1': 128, 'h2': 64,  'drop': 0.3, 'lr': 0.01},
    {'h1': 128, 'h2': 128, 'drop': 0.2, 'lr': 0.01},
    {'h1': 128, 'h2': 128, 'drop': 0.3, 'lr': 0.01},
]

tune_results = []
skf3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
for i, c in enumerate(configs):
    print(f"Config {i+1}/{len(configs)}: {c}")
    f1s = []
    for tr_i, val_i in skf3.split(X_train_scaled, y_train):
        m = build_keras_model(c['h1'], c['h2'], c['drop'], c['lr'])
        m.fit(X_train_scaled.values[tr_i], y_train.values[tr_i].astype(float),
              epochs=80, batch_size=32, verbose=0,
              validation_data=(X_train_scaled.values[val_i], y_train.values[val_i].astype(float)),
              callbacks=[callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
        p = (m.predict(X_train_scaled.values[val_i], verbose=0).flatten() >= 0.5).astype(int)
        f1s.append(f1_score(y_train.values[val_i], p))
    tune_results.append({**c, 'mean_f1': np.mean(f1s), 'std_f1': np.std(f1s)})
    print(f"  CV F1: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")

tune_df = pd.DataFrame(tune_results).sort_values('mean_f1', ascending=False)
print("\nAll tuning results:")
print(tune_df.to_string(index=False))

# Tuning plot
fig, ax = plt.subplots(figsize=(14, 6))
labels = [f"({int(r['h1'])},{int(r['h2'])})\nD={r['drop']}\nLR={r['lr']}" for _, r in tune_df.iterrows()]
colors = ['#e74c3c' if i == 0 else '#3498db' for i in range(len(tune_df))]
ax.bar(range(len(labels)), tune_df['mean_f1'], yerr=tune_df['std_f1'],
       color=colors, edgecolor='black', linewidth=0.5, capsize=3)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
ax.set_ylabel('Mean CV F1 Score')
ax.set_title('Keras MLP Hyperparameter Tuning (Best in Red)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('mlp_tuning_results.png', dpi=150, bbox_inches='tight')
plt.show()

# Retrain best
best = tune_df.iloc[0]
print(f"\nBest config: H=({int(best['h1'])},{int(best['h2'])}), D={best['drop']}, LR={best['lr']}")
keras_tuned = build_keras_model(int(best['h1']), int(best['h2']), best['drop'], best['lr'])
keras_tuned.fit(
    X_train_scaled.values, y_train.values.astype(float),
    epochs=100, batch_size=32, validation_split=0.15, verbose=1,
    callbacks=[callbacks.EarlyStopping(patience=15, restore_best_weights=True)]
)

y_prob_t = keras_tuned.predict(X_test_scaled.values).flatten()
y_pred_t = (y_prob_t >= 0.5).astype(int)
keras_tuned_m = {
    'Model': 'Neural Network (Keras)',
    'Accuracy': accuracy_score(y_test, y_pred_t),
    'Precision': precision_score(y_test, y_pred_t),
    'Recall': recall_score(y_test, y_pred_t),
    'F1': f1_score(y_test, y_pred_t),
    'AUC-ROC': roc_auc_score(y_test, y_prob_t)
}
print(f"Tuned F1: {keras_tuned_m['F1']:.4f} vs Base F1: {keras_base['F1']:.4f}")

if keras_tuned_m['F1'] >= keras_base['F1']:
    all_results.append(keras_tuned_m)
    keras_tuned.save('model_mlp.keras')
    print("Tuned model saved!")
else:
    all_results.append(keras_base)
    keras_model.save('model_mlp.keras')
    print("Base model saved (it was better)!")

best_params_dict['MLP'] = {
    'hidden_layers': (int(best['h1']), int(best['h2'])),
    'dropout_rate': float(best['drop']),
    'learning_rate': float(best['lr']),
    'optimizer': 'Adam', 'loss': 'binary_crossentropy'
}

# ============================================================
# SAVE METADATA & RESULTS
# ============================================================
metadata = {
    'feature_names': list(X.columns),
    'num_cols': num_cols,
    'categorical_mappings': {
        'Sex': ['F', 'M'], 'ChestPainType': ['ASY', 'ATA', 'NAP', 'TA'],
        'RestingECG': ['LVH', 'Normal', 'ST'], 'ExerciseAngina': ['N', 'Y'],
        'ST_Slope': ['Down', 'Flat', 'Up']
    },
    'best_params': best_params_dict
}
joblib.dump(metadata, 'metadata.joblib')

results_df = pd.DataFrame(all_results).round(4)
results_df.to_csv('model_comparison.csv', index=False)

print(f"\n{'='*60}")
print("  DONE! ALL FILES GENERATED")
print(f"{'='*60}")
print(results_df.to_string(index=False))
print(f"\n🏆 Best: {results_df.loc[results_df['F1'].idxmax(), 'Model']}")
print("""
Download these files from Colab:
  model_logistic_regression.joblib
  model_decision_tree.joblib
  model_random_forest.joblib
  model_xgboost.joblib
  model_mlp.keras
  scaler.joblib
  metadata.joblib
  model_comparison.csv
  mlp_training_history.png
  mlp_tuning_results.png
""")
