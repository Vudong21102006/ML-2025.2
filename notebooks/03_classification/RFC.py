"""
=======================================================================
  RFC — Random Forest Classification (Wine Quality Dataset)
=======================================================================
  Mục tiêu : Phân loại chất lượng rượu: 0=low / 1=medium / 2=high
  Data      : data/processed/classification/train_clf_unscaled.csv
              (Unscaled — tree-based models không cần StandardScaling)
  Tuning    : RandomizedSearchCV (n_estimators, max_depth, ...)

  Cách chạy (từ thư mục gốc dự án):
      python notebooks/03_classification/RFC.py

  Output:
      models/rfc_best.pkl
      reports/results/rfc_results.csv
      reports/figures/rfc_confusion_matrix.png
      reports/figures/rfc_feature_importance.png
      reports/figures/rfc_n_estimators_curve.png
      reports/figures/rfc_proba_dist.png
=======================================================================
"""

# ── 0. Import ─────────────────────────────────────────────────────────
import sys, os

NOTEBOOKS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, NOTEBOOKS_DIR)
from utils import *

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

SCRIPT = "RFC"
print_section(f"SCRIPT: {SCRIPT}  —  Random Forest Classification")
print(f"  Project dir: {BASE}")
print(f"  Data dir   : {CLF_DIR}  (Unscaled)")
print(f"  Output     : models/  +  reports/results/  +  reports/figures/")


# ═══════════════════════════════════════════════════════════════════════
#  BƯỚC 1: LOAD DATA
# ═══════════════════════════════════════════════════════════════════════
print_section("BƯỚC 1: Load Data")

# RFC: dùng Unscaled data (tree không cần scaling)
X_train, y_train, X_val, y_val, X_test, y_test = load_classification_data_unscaled()
feat_names = list(X_train.columns)

print(f"  Train : {X_train.shape}  |  Val : {X_val.shape}  |  Test : {X_test.shape}")
print(f"\n  Phân bố lớp trong Train set:")
for i, lbl in enumerate(LABEL_NAMES):
    cnt = int((y_train == i).sum())
    print(f"    {i} ({lbl:>6}) : {cnt:>4}  ({cnt/len(y_train)*100:.1f}%)")
print(f"\n  → class_weight='balanced' sẽ tự điều chỉnh trọng số nghịch đảo tần suất lớp")


# ═══════════════════════════════════════════════════════════════════════
#  BƯỚC 2: BASELINE
# ═══════════════════════════════════════════════════════════════════════
print_section("BƯỚC 2: Baseline RFC (class_weight='balanced', n=100)")

t0 = time.time()
rfc_base = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",  # Trọng số tỷ lệ nghịch với tần suất lớp:
                               # w_i = n_samples / (n_classes * n_samples_i)
                               # → lớp 'low' (4.5%) được trọng số cao hơn nhiều
    random_state=42,
    n_jobs=-1,
    oob_score=True,
)
rfc_base.fit(X_train, y_train)
print(f"  Thời gian train: {time.time() - t0:.1f}s")
print(f"  OOB Score      : {rfc_base.oob_score_:.4f}")

print("\n  Baseline performance:")
evaluate_classification(y_train, rfc_base.predict(X_train), split_name="Train", verbose=False)
evaluate_classification(y_val,   rfc_base.predict(X_val),   split_name="Val  ")


# ═══════════════════════════════════════════════════════════════════════
#  BƯỚC 3: HYPERPARAMETER TUNING — RandomizedSearchCV
#
#  Tham số giống RFR nhưng scoring="f1_macro" vì bài toán classification
#  có mất cân bằng lớp → cần metric phản ánh đúng hiệu suất trên cả 3 lớp
# ═══════════════════════════════════════════════════════════════════════
print_section("BƯỚC 3: RandomizedSearchCV (n_iter=40, cv=5, scoring=f1_macro)")

param_dist = {
    "n_estimators"     : [100, 200, 300, 500],
    "max_depth"        : [None, 5, 10, 15, 20],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf" : [1, 2, 4, 8],
    "max_features"     : ["sqrt", "log2", 0.5, 0.7],
}

t0 = time.time()
rand_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(
        class_weight="balanced",   # Fix cứng — không tune
        random_state=42, n_jobs=-1,
    ),
    param_distributions=param_dist,
    n_iter=40,
    cv=5,
    scoring="f1_macro",    # Tối ưu F1-macro (tốt cho imbalanced data)
    n_jobs=-1,
    verbose=1,
    random_state=42,
    refit=True,
)
rand_search.fit(X_train, y_train)

print(f"\n  Thời gian RandomSearch: {time.time() - t0:.1f}s")
print(f"  Best params       : {rand_search.best_params_}")
print(f"  Best CV F1-macro  : {rand_search.best_score_:.4f}")

# In top 5
cv_results = pd.DataFrame(rand_search.cv_results_)
top5 = cv_results.nlargest(5, "mean_test_score")[
    ["params", "mean_test_score", "std_test_score"]
]
print("\n  Top 5 param combinations:")
print(top5.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════
#  BƯỚC 4: ĐÁNH GIÁ MODEL TỐT NHẤT
# ═══════════════════════════════════════════════════════════════════════
print_section("BƯỚC 4: Đánh giá best RFC model")

rfc_best = rand_search.best_estimator_

results = {}
for split, X, y in [("Train", X_train, y_train),
                     ("Val",   X_val,   y_val),
                     ("Test",  X_test,  y_test)]:
    results[split] = evaluate_classification(y, rfc_best.predict(X), split_name=split)


# ═══════════════════════════════════════════════════════════════════════
#  BƯỚC 5: CROSS-VALIDATION trên Train+Val
# ═══════════════════════════════════════════════════════════════════════
print_section("BƯỚC 5: Cross-Validation (5-fold) trên Train+Val")

X_tv = pd.concat([X_train, X_val], ignore_index=True)
y_tv = np.concatenate([y_train, y_val])

cv_f1  = cross_val_score(rfc_best, X_tv, y_tv, cv=5, scoring="f1_macro",  n_jobs=-1)
cv_acc = cross_val_score(rfc_best, X_tv, y_tv, cv=5, scoring="accuracy",   n_jobs=-1)

print(f"  CV F1-macro : {cv_f1.mean():.4f}  ±  {cv_f1.std():.4f}")
print(f"  CV Accuracy : {cv_acc.mean():.4f}  ±  {cv_acc.std():.4f}")


# ═══════════════════════════════════════════════════════════════════════
#  BƯỚC 6: FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════
print_section("BƯỚC 6: Feature Importance (Gini-based / MDI)")

imp_df = pd.DataFrame({
    "feature"   : feat_names,
    "importance": rfc_best.feature_importances_,
}).sort_values("importance", ascending=False)

print("\n  Feature importances (sorted):")
print(imp_df.to_string(index=False))

plot_feature_importance(
    rfc_best.feature_importances_, feat_names,
    save_path=os.path.join(FIG_DIR, "rfc_feature_importance.png"),
    title="RFC — Feature Importance (Gini / MDI)"
)


# ═══════════════════════════════════════════════════════════════════════
#  BƯỚC 7: BIỂU ĐỒ
# ═══════════════════════════════════════════════════════════════════════
print_section("BƯỚC 7: Vẽ biểu đồ")

# a) Confusion Matrix (Test set)
plot_confusion_matrix(
    y_test, rfc_best.predict(X_test),
    save_path=os.path.join(FIG_DIR, "rfc_confusion_matrix.png"),
    title="Random Forest Classification — Confusion Matrix (Test set)"
)

# b) n_estimators vs Val F1-macro — sensitivity
best_p = rand_search.best_params_
n_est_vals = [10, 30, 50, 100, 150, 200, 300, 500]
train_f1s, val_f1s = [], []

for n in n_est_vals:
    m = RandomForestClassifier(
        n_estimators=n,
        max_depth=best_p.get("max_depth"),
        min_samples_split=best_p.get("min_samples_split"),
        min_samples_leaf=best_p.get("min_samples_leaf"),
        max_features=best_p.get("max_features"),
        class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    m.fit(X_train, y_train)
    train_f1s.append(f1_score(y_train, m.predict(X_train), average="macro", zero_division=0))
    val_f1s.append(f1_score(y_val,   m.predict(X_val),   average="macro", zero_division=0))

fig_n, ax_n = plt.subplots(figsize=(9, 5))
ax_n.plot(n_est_vals, train_f1s, "o-",  label="Train F1-macro", color="#5b8ec4", linewidth=2)
ax_n.plot(n_est_vals, val_f1s,   "s--", label="Val F1-macro",   color="#e07b54", linewidth=2)
ax_n.axvline(best_p.get("n_estimators", 100), color="gray", linestyle=":", alpha=0.7,
             label=f"Best n={best_p.get('n_estimators', 100)}")
ax_n.set_xlabel("n_estimators", fontsize=11)
ax_n.set_ylabel("F1-macro", fontsize=11)
ax_n.set_title("RFC — Train vs Val F1-macro theo số cây (n_estimators)",
               fontsize=12, fontweight="bold")
ax_n.legend()
fig_n.tight_layout()
fig_n.savefig(os.path.join(FIG_DIR, "rfc_n_estimators_curve.png"), dpi=130, bbox_inches="tight")
plt.close()
print(f"  Plot saved: rfc_n_estimators_curve.png")

# c) Phân phối xác suất dự đoán cho 3 lớp (Test set)
proba = rfc_best.predict_proba(X_test)   # shape: (798, 3)
colors = ["#e07b54", "#5b8ec4", "#6ec47a"]
fig_p, axes_p = plt.subplots(1, 3, figsize=(14, 4))
fig_p.suptitle("RFC — Phân phối xác suất dự đoán (Test set)", fontsize=12, fontweight="bold")
for i, (ax, lbl, c) in enumerate(zip(axes_p, LABEL_NAMES, colors)):
    ax.hist(proba[:, i], bins=30, color=c, edgecolor="white", alpha=0.85)
    ax.set_title(f"P(class='{lbl}')", fontsize=10)
    ax.set_xlabel("Predicted Probability"); ax.set_ylabel("Count")
fig_p.tight_layout()
fig_p.savefig(os.path.join(FIG_DIR, "rfc_proba_dist.png"), dpi=130, bbox_inches="tight")
plt.close()
print(f"  Plot saved: rfc_proba_dist.png")


# ═══════════════════════════════════════════════════════════════════════
#  BƯỚC 8: LƯU MODEL VÀ KẾT QUẢ
# ═══════════════════════════════════════════════════════════════════════
print_section("BƯỚC 8: Lưu model và kết quả")

save_model(rfc_best, "rfc_best.pkl")
save_results(results, "rfc_results.csv")


# ═══════════════════════════════════════════════════════════════════════
#  TỔNG KẾT
# ═══════════════════════════════════════════════════════════════════════
print_section("TỔNG KẾT — RFC")

print(f"\n  Best hyperparameters:")
for k, v in rand_search.best_params_.items():
    print(f"    {k:<22}: {v}")
print(f"    {'class_weight':<22}: balanced  (fixed)")

print(f"\n  ┌───────────────┬───────────┬───────────┬────────────┐")
print(f"  │     Split     │ Accuracy  │ F1-macro  │ F1-weighted│")
print(f"  ├───────────────┼───────────┼───────────┼────────────┤")
for sp, m in results.items():
    print(f"  │  {sp:<12} │  {m['Accuracy']:.4f}   │  {m['F1_macro']:.4f}   │   {m['F1_weighted']:.4f}   │")
print(f"  └───────────────┴───────────┴───────────┴────────────┘")

print("\n  CV (5-fold, Train+Val):")
print(f"    F1-macro = {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
print(f"    Accuracy = {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")

print("\n  Files:")
print("    models/rfc_best.pkl")
print("    reports/results/rfc_results.csv")
print("    reports/figures/rfc_confusion_matrix.png")
print("    reports/figures/rfc_feature_importance.png")
print("    reports/figures/rfc_n_estimators_curve.png")
print("    reports/figures/rfc_proba_dist.png")

print_section("DONE — RFC.py hoàn thành!")
