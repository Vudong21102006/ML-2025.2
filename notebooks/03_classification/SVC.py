"""
=======================================================================
  SVC — Support Vector Classification (Wine Quality Dataset)
=======================================================================
  Mục tiêu : Phân loại chất lượng rượu: 0=low / 1=medium / 2=high
  Data      : data/processed/classification/train_clf.csv (StandardScaled)
  Tuning    : GridSearchCV (C, kernel, gamma) — scoring=f1_macro

  Cách chạy (từ thư mục gốc dự án):
      python notebooks/03_classification/SVC.py

  Output:
      models/svc_best.pkl
      reports/results/svc_results.csv
      reports/figures/svc_confusion_matrix.png
      reports/figures/svc_C_curve.png
      reports/figures/svc_permutation_importance.png
=======================================================================
"""

# ── 0. Import ─────────────────────────────────────────────────────────
import sys, os

# __file__ = .../notebooks/03_classification/SVC.py
# parent   = .../notebooks/
NOTEBOOKS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, NOTEBOOKS_DIR)
from utils import *

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.inspection import permutation_importance

SCRIPT = "SVC"
print_section(f"SCRIPT: {SCRIPT}  —  Support Vector Classification")
print(f"  Project dir: {BASE}")
print(f"  Data dir   : {CLF_DIR}")
print(f"  Output     : models/  +  reports/results/  +  reports/figures/")


# ═══════════════════════════════════════════════════════════════════════
#  BƯỚC 1: LOAD DATA
# ═══════════════════════════════════════════════════════════════════════
print_section("BƯỚC 1: Load Data")

# SVC cần StandardScaled data
X_train, y_train, X_val, y_val, X_test, y_test = load_classification_data()
feat_names = list(X_train.columns)

print(f"  Train : {X_train.shape}  |  Val : {X_val.shape}  |  Test : {X_test.shape}")
print(f"\n  Phân bố lớp trong Train set:")
for i, lbl in enumerate(LABEL_NAMES):
    cnt = int((y_train == i).sum())
    print(f"    {i} ({lbl:>6}) : {cnt:>4}  ({cnt/len(y_train)*100:.1f}%)")
print(f"\n  → class_weight='balanced' sẽ tự điều chỉnh trọng số cho lớp thiểu số")


# ═══════════════════════════════════════════════════════════════════════
#  BƯỚC 2: BASELINE SVC
# ═══════════════════════════════════════════════════════════════════════
print_section("BƯỚC 2: Baseline SVC (default + class_weight='balanced')")

t0 = time.time()
svc_base = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    class_weight="balanced",  # Quan trọng: xử lý mất cân bằng lớp
    random_state=42,
)
svc_base.fit(X_train, y_train)
print(f"  Thời gian train: {time.time() - t0:.1f}s")

print("\n  Baseline performance:")
evaluate_classification(y_train, svc_base.predict(X_train), split_name="Train", verbose=False)
evaluate_classification(y_val,   svc_base.predict(X_val),   split_name="Val  ")


# ═══════════════════════════════════════════════════════════════════════
#  BƯỚC 3: HYPERPARAMETER TUNING — GridSearchCV
#
#  Các tham số cần tune:
#    C       : Hệ số phạt. C lớn → ranh giới phức tạp hơn (overfit).
#                C nhỏ → ranh giới mềm hơn (regularization mạnh).
#    kernel  : rbf (phổ biến) hoặc poly (đa thức bậc cao).
#    gamma   : Phạm vi ảnh hưởng của mỗi điểm train.
#                'scale' = 1/(n_features * X.var())  ← thường tốt hơn
#                'auto'  = 1/n_features
#
#  NOTE: class_weight='balanced' được fix cứng — luôn bật
#        vì data wine quality có lớp 'low' chỉ 4.5%
# ═══════════════════════════════════════════════════════════════════════
print_section("BƯỚC 3: GridSearchCV (C, kernel, gamma) — cv=5, scoring=f1_macro")

param_grid = {
    "C"     : [0.1, 1, 10, 50, 100],
    "kernel": ["rbf", "poly"],
    "gamma" : ["scale", "auto"],
}
# Tổng: 5 × 2 × 2 = 20 tổ hợp × 5 folds = 100 lần fit

t0 = time.time()
grid = GridSearchCV(
    estimator=SVC(
        class_weight="balanced",  # Fix cứng — không tune tham số này
        random_state=42,
        probability=True,         # Bật để có thể gọi predict_proba() sau
    ),
    param_grid=param_grid,
    cv=5,
    scoring="f1_macro",     # Tối ưu F1-macro (phù hợp imbalanced data)
    n_jobs=-1,
    verbose=1,
    refit=True,
)
grid.fit(X_train, y_train)

print(f"\n  Thời gian GridSearch: {time.time() - t0:.1f}s")
print(f"  Best params      : {grid.best_params_}")
print(f"  Best CV F1-macro : {grid.best_score_:.4f}")

# In top 5 kết quả
cv_results = pd.DataFrame(grid.cv_results_)
top5 = cv_results.nlargest(5, "mean_test_score")[
    ["params", "mean_test_score", "std_test_score", "rank_test_score"]
]
print("\n  Top 5 param combinations:")
print(top5.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════
#  BƯỚC 4: ĐÁNH GIÁ MODEL TỐT NHẤT
# ═══════════════════════════════════════════════════════════════════════
print_section("BƯỚC 4: Đánh giá best SVC model")

svc_best = grid.best_estimator_

results = {}
for split, X, y in [("Train", X_train, y_train),
                     ("Val",   X_val,   y_val),
                     ("Test",  X_test,  y_test)]:
    results[split] = evaluate_classification(y, svc_best.predict(X), split_name=split)


# ═══════════════════════════════════════════════════════════════════════
#  BƯỚC 5: CROSS-VALIDATION trên Train+Val
# ═══════════════════════════════════════════════════════════════════════
print_section("BƯỚC 5: Cross-Validation (5-fold) trên Train+Val")

X_tv = pd.concat([X_train, X_val], ignore_index=True)
y_tv = np.concatenate([y_train, y_val])

cv_f1  = cross_val_score(svc_best, X_tv, y_tv, cv=5, scoring="f1_macro",  n_jobs=-1)
cv_acc = cross_val_score(svc_best, X_tv, y_tv, cv=5, scoring="accuracy",   n_jobs=-1)

print(f"  CV F1-macro : {cv_f1.mean():.4f}  ±  {cv_f1.std():.4f}")
print(f"  CV Accuracy : {cv_acc.mean():.4f}  ±  {cv_acc.std():.4f}")


# ═══════════════════════════════════════════════════════════════════════
#  BƯỚC 6: PERMUTATION FEATURE IMPORTANCE
#  SVM không có feature_importances_ như tree-based models
#  → Dùng Permutation Importance: tắt dần từng feature và đo độ giảm score
# ═══════════════════════════════════════════════════════════════════════
print_section("BƯỚC 6: Permutation Feature Importance (trên Val set)")

perm = permutation_importance(
    svc_best, X_val, y_val,
    n_repeats=15,         # Shuffle mỗi feature 15 lần → kết quả ổn định hơn
    random_state=42,
    scoring="f1_macro",
    n_jobs=-1,
)

# Sắp xếp và in kết quả
perm_df = pd.DataFrame({
    "feature"   : feat_names,
    "importance": perm.importances_mean,
    "std"       : perm.importances_std,
}).sort_values("importance", ascending=False)

print("\n  Top 10 features quan trọng nhất:")
print(perm_df.head(10).to_string(index=False))

plot_feature_importance(
    perm.importances_mean, feat_names,
    save_path=os.path.join(FIG_DIR, "svc_permutation_importance.png"),
    title="SVC — Permutation Feature Importance (Val set, F1-macro)"
)


# ═══════════════════════════════════════════════════════════════════════
#  BƯỚC 7: BIỂU ĐỒ
# ═══════════════════════════════════════════════════════════════════════
print_section("BƯỚC 7: Vẽ biểu đồ")

# a) Confusion Matrix (Test set)
plot_confusion_matrix(
    y_test, svc_best.predict(X_test),
    save_path=os.path.join(FIG_DIR, "svc_confusion_matrix.png"),
    title=(f"SVC (C={grid.best_params_['C']}, "
           f"kernel={grid.best_params_['kernel']}) — Confusion Matrix (Test)")
)

# b) Sensitivity: C vs F1-macro
best_kernel = grid.best_params_["kernel"]
best_gamma  = grid.best_params_["gamma"]

C_vals = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 200]
train_f1s, val_f1s = [], []
for c in C_vals:
    m = SVC(kernel=best_kernel, C=c, gamma=best_gamma,
            class_weight="balanced", random_state=42)
    m.fit(X_train, y_train)
    train_f1s.append(f1_score(y_train, m.predict(X_train), average="macro", zero_division=0))
    val_f1s.append(f1_score(y_val,   m.predict(X_val),   average="macro", zero_division=0))

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(C_vals, train_f1s, "o-",  label="Train F1-macro", color="#5b8ec4", linewidth=2)
ax.plot(C_vals, val_f1s,   "s--", label="Val F1-macro",   color="#e07b54", linewidth=2)
ax.axvline(grid.best_params_["C"], color="gray", linestyle=":", alpha=0.7,
           label=f"Best C={grid.best_params_['C']}")
ax.set_xscale("log")
ax.set_xlabel("C (log scale)", fontsize=11)
ax.set_ylabel("F1-macro", fontsize=11)
ax.set_title(f"SVC — Train vs Val F1-macro theo C  (kernel={best_kernel})",
             fontsize=12, fontweight="bold")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "svc_C_curve.png"), dpi=130, bbox_inches="tight")
plt.close()
print(f"  Plot saved: svc_C_curve.png")


# ═══════════════════════════════════════════════════════════════════════
#  BƯỚC 8: LƯU MODEL VÀ KẾT QUẢ
# ═══════════════════════════════════════════════════════════════════════
print_section("BƯỚC 8: Lưu model và kết quả")

save_model(svc_best, "svc_best.pkl")
save_results(results, "svc_results.csv")


# ═══════════════════════════════════════════════════════════════════════
#  TỔNG KẾT
# ═══════════════════════════════════════════════════════════════════════
print_section("TỔNG KẾT — SVC")

print(f"\n  Best hyperparameters:")
for k, v in grid.best_params_.items():
    print(f"    {k:<10}: {v}")
print(f"    class_weight: balanced  (fixed)")

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
print("    models/svc_best.pkl")
print("    reports/results/svc_results.csv")
print("    reports/figures/svc_confusion_matrix.png")
print("    reports/figures/svc_C_curve.png")
print("    reports/figures/svc_permutation_importance.png")

print_section("DONE — SVC.py hoàn thành!")
