"""
train_models.py
───────────────
Standalone training script that:
  1. Loads CSV datasets
  2. Cleans & engineers features
  3. Trains XGBoost models for Diabetes, Heart Disease, Liver Disease
  4. Saves models, scalers, and encoders to ../models/

Run from the notebooks/ directory:
    python train_models.py
Or from the project root:
    python notebooks/train_models.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc,
    classification_report,
)

warnings.filterwarnings("ignore")

# ── Resolve paths regardless of working directory ──────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
MODEL_DIR   = os.path.join(PROJECT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Helper ─────────────────────────────────────────────────────────────────────
def save_fig(name: str):
    path = os.path.join(MODEL_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {name}")


def eval_model(y_true, y_pred, y_prob, label: str):
    """Print evaluation metrics and return dict."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    print(f"\n  {'─'*40}")
    print(f"  {label} — Evaluation")
    print(f"  {'─'*40}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC AUC   : {roc_auc:.4f}")
    print("\n" + classification_report(y_true, y_pred))
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, roc_auc=roc_auc,
                fpr=fpr, tpr=tpr)


def plot_cm_and_importance(cm, model, feature_names, label: str, cmap: str, fname: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Confusion matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=axes[0],
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    axes[0].set_title(f"{label} – Confusion Matrix", fontweight="bold")
    axes[0].set_ylabel("Actual"); axes[0].set_xlabel("Predicted")
    # Feature importance
    imp_df = pd.DataFrame({"Feature": feature_names,
                           "Importance": model.feature_importances_}
                         ).sort_values("Importance", ascending=True)
    colours = plt.cm.viridis(np.linspace(0.2, 0.9, len(imp_df)))
    axes[1].barh(imp_df["Feature"], imp_df["Importance"], color=colours)
    axes[1].set_title(f"{label} – Feature Importance", fontweight="bold")
    axes[1].set_xlabel("Importance Score")
    save_fig(fname)


# ══════════════════════════════════════════════════════════════════════════════
# DISEASE 1 — DIABETES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  DIABETES")
print("═"*60)

df_d = pd.read_csv(os.path.join(DATA_DIR, "diabetes.csv"))
print(f"  Loaded: {df_d.shape}")

# --- Cleaning ---
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in zero_cols:
    n = (df_d[col] == 0).sum()
    if n:
        df_d[col] = df_d[col].replace(0, np.nan)
        df_d[col].fillna(df_d[col].median(), inplace=True)
        print(f"  [CLEAN] {col}: {n} zeros → imputed with median")

# --- Feature Engineering ---
df_d["BMI_Category"] = pd.cut(df_d["BMI"],
    bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]).astype(int)
df_d["Age_Group"] = pd.cut(df_d["Age"],
    bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3]).astype(int)
df_d["Glucose_Insulin_Ratio"] = df_d["Glucose"] / (df_d["Insulin"] + 1)

# --- Correlation heatmap ---
plt.figure(figsize=(12, 8))
mask = np.triu(np.ones_like(df_d.corr(), dtype=bool))
sns.heatmap(df_d.corr(), mask=mask, annot=True, fmt=".2f",
            cmap="RdYlGn", center=0, linewidths=0.5)
plt.title("Diabetes – Correlation Heatmap", fontsize=14, fontweight="bold")
save_fig("diabetes_heatmap.png")

# --- Train ---
X_d = df_d.drop("Outcome", axis=1)
y_d = df_d["Outcome"]
scaler_d = StandardScaler()
X_ds = scaler_d.fit_transform(X_d)
Xt_d, Xv_d, yt_d, yv_d = train_test_split(
    X_ds, y_d, test_size=0.2, random_state=42, stratify=y_d)

model_d = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                         subsample=0.8, colsample_bytree=0.8,
                         eval_metric="logloss", random_state=42)
model_d.fit(Xt_d, yt_d)
yp_d  = model_d.predict(Xv_d)
ypr_d = model_d.predict_proba(Xv_d)[:, 1]
metrics_d = eval_model(yv_d, yp_d, ypr_d, "Diabetes")

plot_cm_and_importance(confusion_matrix(yv_d, yp_d), model_d,
    list(X_d.columns), "Diabetes", "Blues", "diabetes_evaluation.png")

# ROC
plt.figure(figsize=(6, 4))
plt.plot(metrics_d["fpr"], metrics_d["tpr"], color="steelblue", lw=2,
         label=f'AUC = {metrics_d["roc_auc"]:.2f}')
plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("Diabetes – ROC Curve"); plt.legend()
save_fig("diabetes_roc.png")

# --- Save ---
joblib.dump(model_d,          os.path.join(MODEL_DIR, "diabetes_model.pkl"))
joblib.dump(scaler_d,         os.path.join(MODEL_DIR, "diabetes_scaler.pkl"))
joblib.dump(list(X_d.columns),os.path.join(MODEL_DIR, "diabetes_features.pkl"))
print("  ✅ Diabetes model saved.")


# ══════════════════════════════════════════════════════════════════════════════
# DISEASE 2 — HEART DISEASE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  HEART DISEASE")
print("═"*60)

df_h = pd.read_csv(os.path.join(DATA_DIR, "heart_disease.csv"))
print(f"  Loaded: {df_h.shape}")

# --- Encoding ---
le_sex  = LabelEncoder(); df_h["sex"]  = le_sex.fit_transform(df_h["sex"])
le_cp   = LabelEncoder(); df_h["cp"]   = le_cp.fit_transform(df_h["cp"])
le_thal = LabelEncoder()
df_h["thal"] = le_thal.fit_transform(df_h["thal"].fillna("normal"))

# --- Impute ---
for col in df_h.columns:
    if df_h[col].isnull().sum():
        df_h[col].fillna(df_h[col].median(), inplace=True)
        print(f"  [IMPUTE] {col} filled with median")

# --- Feature Engineering ---
df_h["age_thalach_ratio"] = df_h["age"] / (df_h["thalach"] + 1)
df_h["bp_chol_index"]     = df_h["trestbps"] * df_h["chol"] / 10000

# --- Heatmap ---
plt.figure(figsize=(14, 9))
mask = np.triu(np.ones_like(df_h.corr(), dtype=bool))
sns.heatmap(df_h.corr(), mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, linewidths=0.5)
plt.title("Heart Disease – Correlation Heatmap", fontsize=14, fontweight="bold")
save_fig("heart_heatmap.png")

# --- Train ---
X_h = df_h.drop("target", axis=1)
y_h = df_h["target"]
scaler_h = StandardScaler()
X_hs = scaler_h.fit_transform(X_h)
Xt_h, Xv_h, yt_h, yv_h = train_test_split(
    X_hs, y_h, test_size=0.2, random_state=42, stratify=y_h)

model_h = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                         subsample=0.8, colsample_bytree=0.8,
                         eval_metric="logloss", random_state=42)
model_h.fit(Xt_h, yt_h)
yp_h  = model_h.predict(Xv_h)
ypr_h = model_h.predict_proba(Xv_h)[:, 1]
metrics_h = eval_model(yv_h, yp_h, ypr_h, "Heart Disease")

plot_cm_and_importance(confusion_matrix(yv_h, yp_h), model_h,
    list(X_h.columns), "Heart Disease", "Reds", "heart_evaluation.png")

plt.figure(figsize=(6, 4))
plt.plot(metrics_h["fpr"], metrics_h["tpr"], color="crimson", lw=2,
         label=f'AUC = {metrics_h["roc_auc"]:.2f}')
plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("Heart Disease – ROC Curve"); plt.legend()
save_fig("heart_roc.png")

joblib.dump(model_h,          os.path.join(MODEL_DIR, "heart_model.pkl"))
joblib.dump(scaler_h,         os.path.join(MODEL_DIR, "heart_scaler.pkl"))
joblib.dump(list(X_h.columns),os.path.join(MODEL_DIR, "heart_features.pkl"))
joblib.dump({"sex": le_sex, "cp": le_cp, "thal": le_thal},
            os.path.join(MODEL_DIR, "heart_encoders.pkl"))
print("  ✅ Heart disease model saved.")


# ══════════════════════════════════════════════════════════════════════════════
# DISEASE 3 — LIVER DISEASE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  LIVER DISEASE")
print("═"*60)

df_l = pd.read_csv(os.path.join(DATA_DIR, "liver_disease.csv"))
print(f"  Loaded: {df_l.shape}")

# --- Fix target ---
df_l["target"] = (df_l["Dataset"] == 1).astype(int)
df_l.drop("Dataset", axis=1, inplace=True)

# --- Encode ---
le_gen = LabelEncoder(); df_l["Gender"] = le_gen.fit_transform(df_l["Gender"])
print(f"  [ENCODE] Gender: {dict(zip(le_gen.classes_, le_gen.transform(le_gen.classes_)))}")

# --- Impute ---
for col in df_l.columns:
    if df_l[col].isnull().sum():
        df_l[col].fillna(df_l[col].median(), inplace=True)
        print(f"  [IMPUTE] {col} filled with median")

# --- Feature Engineering ---
df_l["AST_ALT_Ratio"]       = df_l["Aspartate_Aminotransferase"] / (df_l["Alamine_Aminotransferase"] + 1)
df_l["Bilirubin_Ratio"]     = df_l["Direct_Bilirubin"] / (df_l["Total_Bilirubin"] + 1)
df_l["Protein_Albumin_Diff"] = df_l["Total_Protiens"] - df_l["Albumin"]

# --- Heatmap ---
plt.figure(figsize=(14, 9))
mask = np.triu(np.ones_like(df_l.corr(), dtype=bool))
sns.heatmap(df_l.corr(), mask=mask, annot=True, fmt=".2f",
            cmap="YlOrRd", center=0, linewidths=0.5)
plt.title("Liver Disease – Correlation Heatmap", fontsize=14, fontweight="bold")
save_fig("liver_heatmap.png")

# --- Train ---
X_l = df_l.drop("target", axis=1)
y_l = df_l["target"]
scaler_l = StandardScaler()
X_ls = scaler_l.fit_transform(X_l)
Xt_l, Xv_l, yt_l, yv_l = train_test_split(
    X_ls, y_l, test_size=0.2, random_state=42, stratify=y_l)

spw = (y_l == 0).sum() / max((y_l == 1).sum(), 1)
model_l = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                         subsample=0.8, colsample_bytree=0.8,
                         scale_pos_weight=spw,
                         eval_metric="logloss", random_state=42)
model_l.fit(Xt_l, yt_l)
yp_l  = model_l.predict(Xv_l)
ypr_l = model_l.predict_proba(Xv_l)[:, 1]
metrics_l = eval_model(yv_l, yp_l, ypr_l, "Liver Disease")

plot_cm_and_importance(confusion_matrix(yv_l, yp_l), model_l,
    list(X_l.columns), "Liver Disease", "Oranges", "liver_evaluation.png")

plt.figure(figsize=(6, 4))
plt.plot(metrics_l["fpr"], metrics_l["tpr"], color="darkorange", lw=2,
         label=f'AUC = {metrics_l["roc_auc"]:.2f}')
plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("Liver Disease – ROC Curve"); plt.legend()
save_fig("liver_roc.png")

joblib.dump(model_l,           os.path.join(MODEL_DIR, "liver_model.pkl"))
joblib.dump(scaler_l,          os.path.join(MODEL_DIR, "liver_scaler.pkl"))
joblib.dump(list(X_l.columns), os.path.join(MODEL_DIR, "liver_features.pkl"))
joblib.dump({"Gender": le_gen}, os.path.join(MODEL_DIR, "liver_encoders.pkl"))
print("  ✅ Liver disease model saved.")


# ══════════════════════════════════════════════════════════════════════════════
# ROC COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
plt.figure(figsize=(8, 6))
plt.plot(metrics_d["fpr"], metrics_d["tpr"], color="steelblue", lw=2,
         label=f'Diabetes (AUC={metrics_d["roc_auc"]:.2f})')
plt.plot(metrics_h["fpr"], metrics_h["tpr"], color="crimson", lw=2,
         label=f'Heart Disease (AUC={metrics_h["roc_auc"]:.2f})')
plt.plot(metrics_l["fpr"], metrics_l["tpr"], color="darkorange", lw=2,
         label=f'Liver Disease (AUC={metrics_l["roc_auc"]:.2f})')
plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curve Comparison – All Models", fontsize=14, fontweight="bold")
plt.legend(fontsize=11); plt.grid(alpha=0.3)
save_fig("roc_comparison.png")

summary = pd.DataFrame({
    "Disease":   ["Diabetes", "Heart Disease", "Liver Disease"],
    "Accuracy":  [metrics_d["acc"],  metrics_h["acc"],  metrics_l["acc"]],
    "Precision": [metrics_d["prec"], metrics_h["prec"], metrics_l["prec"]],
    "Recall":    [metrics_d["rec"],  metrics_h["rec"],  metrics_l["rec"]],
    "F1 Score":  [metrics_d["f1"],   metrics_h["f1"],   metrics_l["f1"]],
    "ROC AUC":   [metrics_d["roc_auc"], metrics_h["roc_auc"], metrics_l["roc_auc"]],
}).set_index("Disease").round(4)

print("\n" + "═"*60)
print("  MODEL COMPARISON SUMMARY")
print("═"*60)
print(summary.to_string())
print("\n✅ All models trained and saved successfully!")
print(f"   Models location: {MODEL_DIR}")
