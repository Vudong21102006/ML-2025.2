"""
=======================================================================
  SVR — Support Vector Regression (Wine Quality Dataset)
=======================================================================
  Mục tiêu : Dự đoán điểm chất lượng rượu (continuous: 3 → 9)
  Data      : data/processed/regression/train.csv  (StandardScaled)
  Tuning    : GridSearchCV (C, epsilon, kernel, gamma)

  Cách chạy (từ thư mục gốc dự án):
      python notebooks/02_regression/SVR.py

  Output:
      models/svr_best.pkl
      reports/results/svr_results.csv
      reports/figures/svr_scatter.png
      reports/figures/svr_C_curve.png
=======================================================================
"""

# ── 0. Import ─────────────────────────────────────────────────────────
import sys, os

# Thêm thư mục notebooks vào path để import utils.py
# __file__ = .../notebooks/02_regression/SVR.py
# parent   = .../notebooks/
NOTEBOOKS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, NOTEBOOKS_DIR)
from utils import *                 # load tất cả hàm & hằng số dùng chung

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, cross_val_score

SCRIPT = "SVR"
print_section(f"SCRIPT: {SCRIPT}  —  Support Vector Regression")
print(f"  Project dir: {BASE}")
print(f"  Data dir   : {REG_DIR}")
print(f"  Output     : models/  +  reports/results/  +  reports/figures/")


# ═══════════════════════════════════════════════════════════════════════
#  BƯỚC 1: LOAD DATA
# ═══════════════════════════════════════════════════════════════════════
print_section("BƯỚC 1: Load Data")

# SVR cần StandardScaled data
X_train, y_train, X_val, y_val, X_test, y_test = load_regression_data()
feat_names = list(X_train.columns)

print(f"  Train : {X_train.shape}  |  Val : {X_val.shape}  |  Test : {X_test.shape}")
print(f"  Target range : {y_train.min()} – {y_train.max()}")
print(f"  Features ({len(feat_names)}): {feat_names}")


# ═══════════════════════════════════════════════════════════════════════
#  BƯỚC 2: BASELINE SVR
#  Chạy với tham số mặc định để có điểm tham chiếu trước khi tune
# ═══════════════════════════════════════════════════════════════════════
print_section("BƯỚC 2: Baseline SVR (default params)")

t0 = time.time()
svr_base = SVR(
    kernel="rbf",    # Radial Basis Function — phổ biến nhất
    C=1.0,           # Hệ số phạt mặc định
    epsilon=0.1,     # Vùng "ống" không tính loss
    gamma="scale",   # gamma = 1 / (n_features * X.var())
)
svr_base.fit(X_train, y_train)
print(f"  Thời gian train: {time.time() - t0:.1f}s")

print("\n  Baseline performance:")
evaluate_regression(y_train, svr_base.predict(X_train), split_name="Train")
evaluate_regression(y_val,   svr_base.predict(X_val),   split_name="Val  ")


# ═══════════════════════════════════════════════════════════════════════
#  BƯỚC 3: HYPERPARAMETER TUNING — GridSearchCV
#
#  Các tham số cần tune:
#    C       : Hệ số phạt (regularization). C lớn → fit sát train hơn
#                (nguy cơ overfit). C nhỏ → margin rộng (underfit).
#    epsilon : Bán kính vùng "không phạt". Điểm nằm trong vùng này
#                không đóng góp vào loss. Nhỏ → fit sát hơn.
#    kernel  : Hàm kernel chiếu dữ liệu lên không gian cao chiều hơn.
#                rbf  = exp(-gamma||x-x'||²)  — phổ biến nhất
#                poly = (gamma*<x,x'>+r)^d    — phù hợp data có bậc cao
#    gamma   : Ảnh hưởng phạm vi của một điểm train ("scale" = tự động)
# ═══════════════════════════════════════════════════════════════════════
print_section("BƯỚC 3: GridSearchCV (C, epsilon, kernel, gamma) — cv=5")

param_grid = {
    "C"      : [0.1, 1, 10, 100],
    "epsilon": [0.05, 0.1, 0.2, 0.5],
    "kernel" : ["rbf", "poly"],
    "gamma"  : ["scale", "auto"],
}
# Tổng: 4 × 4 × 2 × 2 = 64 tổ hợp × 5 folds = 320 lần fit

t0 = time.time()
grid = GridSearchCV(
    estimator=SVR(),
    param_grid=param_grid,
    cv=5,              # 5-fold cross-validation trên train set
    scoring="r2",      # Tối ưu R² (coefficient of determination)
    n_jobs=-1,         # Dùng tất cả CPU cores (song song)
    verbose=1,         # In tiến trình
    refit=True,        # Tự fit lại best model trên toàn train sau khi tìm được
)
grid.fit(X_train, y_train)

print(f"\n  Thời gian GridSearch: {time.time() - t0:.1f}s")
print(f"  Best params  : {grid.best_params_}")
print(f"  Best CV R²   : {grid.best_score_:.4f}")

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
print_section("BƯỚC 4: Đánh giá best SVR model")

svr_best = grid.best_estimator_

results = {}
for split, X, y in [("Train", X_train, y_train),
                     ("Val",   X_val,   y_val),
                     ("Test",  X_test,  y_test)]:
    results[split] = evaluate_regression(y, svr_best.predict(X), split_name=split)

# Nhận xét overfitting / underfitting
train_r2 = results["Train"]["R2"]
val_r2   = results["Val"]["R2"]
gap      = train_r2 - val_r2
print(f"\n  Train–Val R² gap : {gap:.4f}", end="  ")
if gap > 0.15:
    print("→ Dấu hiệu OVERFIT (thử tăng regularization hoặc giảm C)")
elif val_r2 < 0.3:
    print("→ Dấu hiệu UNDERFIT (thử tăng C hoặc đổi kernel)")
else:
    print("→ Model cân bằng tốt ✓")


# ═══════════════════════════════════════════════════════════════════════
#  BƯỚC 5: CROSS-VALIDATION trên Train+Val
#  Đánh giá thêm bằng cách dùng toàn bộ train+val để cho kết quả ổn định
# ═══════════════════════════════════════════════════════════════════════
print_section("BƯỚC 5: Cross-Validation (5-fold) trên Train+Val")

X_tv = pd.concat([X_train, X_val], ignore_index=True)
y_tv = np.concatenate([y_train, y_val])

cv_r2   = cross_val_score(svr_best, X_tv, y_tv, cv=5, scoring="r2", n_jobs=-1)
cv_rmse = cross_val_score(svr_best, X_tv, y_tv, cv=5,
                          scoring="neg_root_mean_squared_error", n_jobs=-1)
cv_mae  = cross_val_score(svr_best, X_tv, y_tv, cv=5,
                          scoring="neg_mean_absolute_error", n_jobs=-1)

print(f"  CV R²   : {cv_r2.mean():.4f}  ±  {cv_r2.std():.4f}")
print(f"  CV RMSE : {(-cv_rmse).mean():.4f}  ±  {(-cv_rmse).std():.4f}")
print(f"  CV MAE  : {(-cv_mae).mean():.4f}  ±  {(-cv_mae).std():.4f}")


# ═══════════════════════════════════════════════════════════════════════
#  BƯỚC 6: BIỂU ĐỒ
# ═══════════════════════════════════════════════════════════════════════
print_section("BƯỚC 6: Vẽ biểu đồ")

# a) Actual vs Predicted + Residual Plot (Test set)
plot_regression_scatter(
    y_test,
    svr_best.predict(X_test),
    save_path=os.path.join(FIG_DIR, "svr_scatter.png"),
    title=(f"SVR (C={grid.best_params_['C']}, "
           f"kernel={grid.best_params_['kernel']}, "
           f"ε={grid.best_params_['epsilon']}) — Actual vs Predicted")
)

# b) Sensitivity: C vs R² (giữ các params khác ở best value)
best_kernel  = grid.best_params_["kernel"]
best_epsilon = grid.best_params_["epsilon"]
best_gamma   = grid.best_params_["gamma"]

C_values = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 200]
train_r2s, val_r2s = [], []

for c in C_values:
    m = SVR(kernel=best_kernel, C=c, epsilon=best_epsilon, gamma=best_gamma)
    m.fit(X_train, y_train)
    train_r2s.append(r2_score(y_train, m.predict(X_train)))
    val_r2s.append(r2_score(y_val,   m.predict(X_val)))

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(C_values, train_r2s, "o-",  label="Train R²", color="#5b8ec4", linewidth=2)
ax.plot(C_values, val_r2s,   "s--", label="Val R²",   color="#e07b54", linewidth=2)
ax.axvline(grid.best_params_["C"], color="gray", linestyle=":", alpha=0.7,
           label=f"Best C={grid.best_params_['C']}")
ax.set_xscale("log")
ax.set_xlabel("C (log scale)", fontsize=11)
ax.set_ylabel("R²", fontsize=11)
ax.set_title(f"SVR — Train vs Val R² theo C  (kernel={best_kernel}, ε={best_epsilon})",
             fontsize=12, fontweight="bold")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "svr_C_curve.png"), dpi=130, bbox_inches="tight")
plt.close()
print(f"  Plot saved: svr_C_curve.png")


# ═══════════════════════════════════════════════════════════════════════
#  BƯỚC 7: LƯU MODEL VÀ KẾT QUẢ
# ═══════════════════════════════════════════════════════════════════════
print_section("BƯỚC 7: Lưu model và kết quả")

save_model(svr_best, "svr_best.pkl")
save_results(results, "svr_results.csv")


# ═══════════════════════════════════════════════════════════════════════
#  TỔNG KẾT
# ═══════════════════════════════════════════════════════════════════════
print_section("TỔNG KẾT — SVR")

print(f"\n  Best hyperparameters:")
for k, v in grid.best_params_.items():
    print(f"    {k:<10}: {v}")

print(f"\n  ┌───────────────┬───────────┬───────────┬───────────┐")
print(f"  │     Split     │    MAE    │   RMSE    │     R²    │")
print(f"  ├───────────────┼───────────┼───────────┼───────────┤")
for sp, m in results.items():
    print(f"  │  {sp:<12} │  {m['MAE']:.4f}   │  {m['RMSE']:.4f}   │  {m['R2']:.4f}   │")
print(f"  └───────────────┴───────────┴───────────┴───────────┘")

print("\n  CV (5-fold, Train+Val):")
print(f"    R²   = {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
print(f"    RMSE = {(-cv_rmse).mean():.4f} ± {(-cv_rmse).std():.4f}")

print("\n  Files:")
print("    models/svr_best.pkl")
print("    reports/results/svr_results.csv")
print("    reports/figures/svr_scatter.png")
print("    reports/figures/svr_C_curve.png")

print_section("DONE — SVR.py hoàn thành!")
