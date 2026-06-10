"""
utils.py — Shared utilities for Wine Quality ML Pipeline
=========================================================
Import trong mỗi script:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils import *
"""

import os, sys, io, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Paths ─────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR    = os.path.join(BASE, "data", "processed")
REG_DIR     = os.path.join(PROC_DIR, "regression")       # Regression splits
CLF_DIR     = os.path.join(PROC_DIR, "classification")   # Classification splits
CLU_DIR     = os.path.join(PROC_DIR, "clustering")       # K-means full dataset
FULL_DIR    = os.path.join(PROC_DIR, "full")              # Full processed datasets
MODEL_DIR   = os.path.join(BASE, "models")
RESULTS_DIR = os.path.join(BASE, "reports", "results")
FIG_DIR     = os.path.join(BASE, "reports", "figures")

for d in [MODEL_DIR, RESULTS_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────
TARGET      = "quality"
LABEL_COL   = "quality_label_enc"      # 0=low, 1=medium, 2=high
LABEL_NAMES = ["low", "medium", "high"]

# ── Data Loaders ──────────────────────────────────────────────────────
def _split_X_y(df, target_col):
    y = df[target_col].values
    X = df.drop(columns=[target_col])
    return X, y

def load_regression_data():
    """StandardScaled + quality. Dung cho: Linear, Ridge, KNN, SVM (SVR), Neural Net."""
    train = pd.read_csv(os.path.join(REG_DIR, "train.csv"))
    val   = pd.read_csv(os.path.join(REG_DIR, "val.csv"))
    test  = pd.read_csv(os.path.join(REG_DIR, "test.csv"))
    return (*_split_X_y(train, TARGET), *_split_X_y(val, TARGET), *_split_X_y(test, TARGET))

def load_regression_data_unscaled():
    """Unscaled + quality. Dung cho: Decision Tree, Random Forest, LightGBM (Regression)."""
    train = pd.read_csv(os.path.join(REG_DIR, "train_unscaled.csv"))
    val   = pd.read_csv(os.path.join(REG_DIR, "val_unscaled.csv"))
    test  = pd.read_csv(os.path.join(REG_DIR, "test_unscaled.csv"))
    return (*_split_X_y(train, TARGET), *_split_X_y(val, TARGET), *_split_X_y(test, TARGET))

def load_classification_data():
    """StandardScaled + quality_label_enc (0/1/2). Dung cho: Logistic, KNN, SVM (SVC), Neural Net."""
    train = pd.read_csv(os.path.join(CLF_DIR, "train_clf.csv"))
    val   = pd.read_csv(os.path.join(CLF_DIR, "val_clf.csv"))
    test  = pd.read_csv(os.path.join(CLF_DIR, "test_clf.csv"))
    return (*_split_X_y(train, LABEL_COL), *_split_X_y(val, LABEL_COL), *_split_X_y(test, LABEL_COL))

def load_classification_data_unscaled():
    """Unscaled + quality_label_enc. Dung cho: Decision Tree, Random Forest, LightGBM+SMOTE."""
    train = pd.read_csv(os.path.join(CLF_DIR, "train_clf_unscaled.csv"))
    val   = pd.read_csv(os.path.join(CLF_DIR, "val_clf_unscaled.csv"))
    test  = pd.read_csv(os.path.join(CLF_DIR, "test_clf_unscaled.csv"))
    return (*_split_X_y(train, LABEL_COL), *_split_X_y(val, LABEL_COL), *_split_X_y(test, LABEL_COL))

def load_kmeans_data():
    """Full dataset 5318 mau, StandardScaled. Dung cho K-means clustering analysis.
    Returns: X (DataFrame, 15 features), y_quality (array 3-9), y_label (array 0/1/2)
    """
    df = pd.read_csv(os.path.join(CLU_DIR, "kmeans_full_scaled.csv"))
    y_quality = df[TARGET].values
    y_label   = df[LABEL_COL].values
    X = df.drop(columns=[TARGET, LABEL_COL])
    return X, y_quality, y_label

# ── Evaluation ────────────────────────────────────────────────────────
def evaluate_regression(y_true, y_pred, split_name="Test", verbose=True):
    """Returns dict {MAE, RMSE, R2}."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    if verbose:
        print(f"  [{split_name}] MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")
    return {"MAE": round(mae,4), "RMSE": round(rmse,4), "R2": round(r2,4)}

def evaluate_classification(y_true, y_pred, split_name="Test", verbose=True):
    """Returns dict {Accuracy, F1_macro, F1_weighted}."""
    acc  = accuracy_score(y_true, y_pred)
    f1m  = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1w  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    if verbose:
        print(f"  [{split_name}] Acc={acc:.4f}  F1-macro={f1m:.4f}  F1-weighted={f1w:.4f}")
        if split_name in ("Val", "Test"):
            print(classification_report(y_true, y_pred,
                                        target_names=LABEL_NAMES, zero_division=0))
    return {"Accuracy": round(acc,4), "F1_macro": round(f1m,4), "F1_weighted": round(f1w,4)}

# ── Model Persistence ─────────────────────────────────────────────────
def save_model(model, filename):
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Model saved: {path}")

def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "rb") as f:
        return pickle.load(f)

# ── Result Persistence ────────────────────────────────────────────────
def save_results(results_dict, filename):
    """results_dict: {split_name: {metric: value}}"""
    rows = []
    for split, metrics in results_dict.items():
        row = {"split": split}
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, filename)
    df.to_csv(path, index=False)
    print(f"  Results saved: {path}")
    return df

# ── Plotting Helpers ──────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.0)

def plot_regression_scatter(y_true, y_pred, save_path, title="Actual vs Predicted"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    lo = min(float(np.min(y_true)), float(np.min(y_pred))) - 0.3
    hi = max(float(np.max(y_true)), float(np.max(y_pred))) + 0.3
    axes[0].scatter(y_true, y_pred, alpha=0.35, s=18, c="#5b8ec4", edgecolors="none")
    axes[0].plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect")
    axes[0].set_xlabel("Actual Quality"); axes[0].set_ylabel("Predicted Quality")
    axes[0].set_title("Actual vs Predicted"); axes[0].legend()
    residuals = np.array(y_pred) - np.array(y_true)
    axes[1].scatter(y_pred, residuals, alpha=0.35, s=18, c="#e07b54", edgecolors="none")
    axes[1].axhline(0, color="red", lw=1.5, linestyle="--")
    axes[1].set_xlabel("Predicted Quality"); axes[1].set_ylabel("Residual")
    axes[1].set_title("Residual Plot")
    plt.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Plot saved: {os.path.basename(save_path)}")

def plot_confusion_matrix(y_true, y_pred, save_path, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Plot saved: {os.path.basename(save_path)}")

def plot_feature_importance(importances, feature_names, save_path,
                            title="Feature Importance", top_n=15):
    importances = np.array(importances)
    idx   = np.argsort(importances)[-top_n:]
    names = np.array(feature_names)[idx]
    vals  = importances[idx]
    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.4)))
    ax.barh(range(len(vals)), vals, color="#5b8ec4", edgecolor="white")
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Importance"); ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Plot saved: {os.path.basename(save_path)}")

def plot_learning_curve(train_scores, val_scores, param_name, param_values, save_path, title=""):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(param_values, train_scores, "o-", label="Train", color="#5b8ec4")
    ax.plot(param_values, val_scores,   "s--", label="Val",  color="#e07b54")
    ax.set_xlabel(param_name); ax.set_ylabel("Score")
    ax.set_title(title or f"Score vs {param_name}", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Plot saved: {os.path.basename(save_path)}")

def print_section(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)
