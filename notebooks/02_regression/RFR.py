"""
=======================================================================
  RFR — Random Forest Regression (Wine Quality Dataset)
=======================================================================
  Mục tiêu : Dự đoán điểm chất lượng rượu (continuous: 3 → 9)
  Data      : data/processed/regression/train_unscaled.csv (Unscaled)
              (Tree-based models không cần StandardScaling)
  Tuning    : RandomizedSearchCV (n_estimators, max_depth, ...)

  Cách chạy (từ thư mục gốc dự án):
      python notebooks/02_regression/RFR.py

  Output:
      models/rfr_best.pkl
      reports/results/rfr_results.csv
      reports/figures/rfr_scatter.png
      reports/figures/rfr_feature_importance.png
      reports/figures/rfr_n_estimators_curve.png
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

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

SCRIPT = "RFR"
print_section(f"SCRIPT: {SCRIPT}  —  Random Forest Regression")
print(f"  Project dir: {BASE}")
print(f"  Data dir   : {REG_DIR}  (Unscaled)")
print(f"  Output     : models/  +  reports/results/  +  reports/figures/")


# ═══════════════════════════════════════════════════════════════════════
#  BƯỚC 1: LOAD DATA
# ═══════════════════════════════════════════════════════════════════════
print_section("BƯỚC 1: Load Data")

# Random Forest: dùng Unscaled data (tree không nhạy với scale)
X_train, y_train, X_val, y_val, X_test, y_test = load_regression_data_unscaled()
feat_names = list(X_train.columns)

print(f"  Train : {X_train.shape}  |  Val : {X_val.shape}  |  Test : {X_test.shape}")
print(f"  Target range : {y_train.min()} – {y_train.max()}")
print(f"  Features ({len(feat_names)}): {feat_names}")


# ═══════════════════════════════════════════════════════════════════════
#  BƯỚC 2: BASELINE
# ═══════════════════════════════════════════════════════════════════════
print_section("BƯỚC 2: Baseline RFR (n_estimators=100, default params)")

t0 = time.time()
rfr_base = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    oob_score=True,      # Out-of-Bag score: đánh giá "miễn phí" không cần val set
                         # Mỗi cây chỉ dùng ~63% samples (bootstrap)
                         # 37% còn lại được dùng để test → ước lượng generalization error
)
rfr_base.fit(X_train, y_train)
print(f"  Thời gian train: {time.time() - t0:.1f}s")
print(f"  OOB R² score   : {rfr_base.oob_score_:.4f}  (estimate không cần val set)")

print("\n  Baseline performance:")
evaluate_regression(y_train, rfr_base.predict(X_train), split_name="Train")
evaluate_regression(y_val,   rfr_base.predict(X_val),   split_name="Val  ")


# ═══════════════════════════════════════════════════════════════════════
#  BƯỚC 3: HYPERPARAMETER TUNING — RandomizedSearchCV
#
#  Dùng RandomizedSearch (không phải GridSearch) vì không gian param lớn:
#  GridSearch đầy đủ sẽ có 4×5×4×4×4 = 1,280 tổ hợp × 5 folds = 6,400 fits
#  RandomizedSearch chỉ lấy ngẫu nhiên n_iter=40 tổ hợp → nhanh ~32x
#
#  Các tham số:
#    n_estimators     : Số cây trong rừng. Nhiều cây → ổn định hơn (giảm variance)
#                       nhưng chậm hơn. Sau ~300 cây thường không cải thiện nhiều.
#    max_depth        : Độ sâu tối đa. None = không giới hạn (cây mọc đến khi
#                       mỗi lá có < min_samples_leaf mẫu). Giới hạn → regularization.
#    min_samples_split: Số mẫu tối thiểu để tiếp tục chia node.
#    min_samples_leaf : Số mẫu tối thiểu ở lá. Tăng → smoother model.
#    max_features     : Số features ngẫu nhiên chọn tại mỗi split.
#                       'sqrt' = sqrt(n_features) — mặc định cho classifier
#                       'log2' = log2(n_features)
#                       float  = tỷ lệ features (0.5 = dùng 50% features)
# ═══════════════════════════════════════════════════════════════════════
print_section("BƯỚC 3: RandomizedSearchCV (n_iter=40, cv=5)")

param_dist = {
    "n_estimators"     : [100, 200, 300, 500],
    "max_depth"        : [None, 5, 10, 15, 20],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf" : [1, 2, 4, 8],
    "max_features"     : ["sqrt", "log2", 0.5, 0.7],
}

t0 = time.time()
rand_search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=40,          # Thử 40 tổ hợp ngẫu nhiên
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=1,
    random_state=42,
    refit=True,
)
rand_search.fit(X_train, y_train)

print(f"\n  Thời gian RandomSearch: {time.time() - t0:.1f}s")
print(f"  Best params  : {rand_search.best_params_}")
print(f"  Best CV R²   : {rand_search.best_score_:.4f}")

# In top 5 kết quả
cv_results = pd.DataFrame(rand_search.cv_results_)
top5 = cv_results.nlargest(5, "mean_test_score")[
    ["params", "mean_test_score", "std_test_score"]
]
print("\n  Top 5 param combinations:")
print(top5.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════
#  BƯỚC 4: ĐÁNH GIÁ MODEL TỐT NHẤT
# ═══════════════════════════════════════════════════════════════════════
print_section("BƯỚC 4: Đánh giá best RFR model")

rfr_best = rand_search.best_estimator_

results = {}
for split, X, y in [("Train", X_train, y_train),
                     ("Val",   X_val,   y_val),
                     ("Test",  X_test,  y_test)]:
    results[split] = evaluate_regression(y, rfr_best.predict(X), split_name=split)

# Overfitting check
gap = results["Train"]["R2"] - results["Val"]["R2"]
print(f"\n  Train–Val R² gap : {gap:.4f}", end="  ")
if gap > 0.20:
    print("→ Dấu hiệu OVERFIT (tăng min_samples_leaf hoặc giảm max_depth)")
elif results["Val"]["R2"] < 0.4:
    print("→ Model yếu (thử tăng n_estimators hoặc thêm features)")
else:
    print("→ Cân bằng tốt ✓")


# ═══════════════════════════════════════════════════════════════════════
#  BƯỚC 5: CROSS-VALIDATION trên Train+Val
# ═══════════════════════════════════════════════════════════════════════
print_section("BƯỚC 5: Cross-Validation (5-fold) trên Train+Val")

X_tv = pd.concat([X_train, X_val], ignore_index=True)
y_tv = np.concatenate([y_train, y_val])

cv_r2   = cross_val_score(rfr_best, X_tv, y_tv, cv=5, scoring="r2", n_jobs=-1)
cv_rmse = cross_val_score(rfr_best, X_tv, y_tv, cv=5,
                          scoring="neg_root_mean_squared_error", n_jobs=-1)
cv_mae  = cross_val_score(rfr_best, X_tv, y_tv, cv=5,
                          scoring="neg_mean_absolute_error", n_jobs=-1)

print(f"  CV R²   : {cv_r2.mean():.4f}  ±  {cv_r2.std():.4f}")
print(f"  CV RMSE : {(-cv_rmse).mean():.4f}  ±  {(-cv_rmse).std():.4f}")
print(f"  CV MAE  : {(-cv_mae).mean():.4f}  ±  {(-cv_mae).std():.4f}")


# ═══════════════════════════════════════════════════════════════════════
#  BƯỚC 6: FEATURE IMPORTANCE
#  Random Forest tính importance dựa trên mức giảm Gini Impurity (MDI)
#  khi split theo mỗi feature, trung bình trên toàn bộ cây
# ═══════════════════════════════════════════════════════════════════════
print_section("BƯỚC 6: Feature Importance (Gini-based / MDI)")

imp_df = pd.DataFrame({
    "feature"   : feat_names,
    "importance": rfr_best.feature_importances_,
}).sort_values("importance", ascending=False)

print("\n  Feature importances (sorted):")
print(imp_df.to_string(index=False))

plot_feature_importance(
    rfr_best.feature_importances_, feat_names,
    save_path=os.path.join(FIG_DIR, "rfr_feature_importance.png"),
    title="RFR — Feature Importance (Gini / MDI)"
)


# ═══════════════════════════════════════════════════════════════════════
#  BƯỚC 7: BIỂU ĐỒ
# ═══════════════════════════════════════════════════════════════════════
print_section("BƯỚC 7: Vẽ biểu đồ")

# a) Actual vs Predicted + Residual (Test set)
plot_regression_scatter(
    y_test, rfr_best.predict(X_test),
    save_path=os.path.join(FIG_DIR, "rfr_scatter.png"),
    title="Random Forest Regression — Actual vs Predicted (Test set)"
)

# b) n_estimators vs Val R² — sensitivity analysis
best_p = rand_search.best_params_
n_est_vals = [10, 30, 50, 100, 150, 200, 300, 500]
train_r2s, val_r2s = [], []

for n in n_est_vals:
    m = RandomForestRegressor(
        n_estimators=n,
        max_depth=best_p.get("max_depth"),
        min_samples_split=best_p.get("min_samples_split"),
        min_samples_leaf=best_p.get("min_samples_leaf"),
        max_features=best_p.get("max_features"),
        random_state=42, n_jobs=-1,
    )
    m.fit(X_train, y_train)
    train_r2s.append(r2_score(y_train, m.predict(X_train)))
    val_r2s.append(r2_score(y_val,   m.predict(X_val)))

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(n_est_vals, train_r2s, "o-",  label="Train R²", color="#5b8ec4", linewidth=2)
ax.plot(n_est_vals, val_r2s,   "s--", label="Val R²",   color="#e07b54", linewidth=2)
ax.axvline(best_p.get("n_estimators", 100), color="gray", linestyle=":", alpha=0.7,
           label=f"Best n={best_p.get('n_estimators', 100)}")
ax.set_xlabel("n_estimators", fontsize=11)
ax.set_ylabel("R²", fontsize=11)
ax.set_title("RFR — Train vs Val R² theo số cây (n_estimators)",
             fontsize=12, fontweight="bold")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "rfr_n_estimators_curve.png"), dpi=130, bbox_inches="tight")
plt.close()
print(f"  Plot saved: rfr_n_estimators_curve.png")


# ═══════════════════════════════════════════════════════════════════════
#  BƯỚC 8: LƯU MODEL VÀ KẾT QUẢ
# ═══════════════════════════════════════════════════════════════════════
print_section("BƯỚC 8: Lưu model và kết quả")

save_model(rfr_best, "rfr_best.pkl")
save_results(results, "rfr_results.csv")


# ═══════════════════════════════════════════════════════════════════════
#  TỔNG KẾT
# ═══════════════════════════════════════════════════════════════════════
print_section("TỔNG KẾT — RFR")

print(f"\n  Best hyperparameters:")
for k, v in rand_search.best_params_.items():
    print(f"    {k:<22}: {v}")

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
print("    models/rfr_best.pkl")
print("    reports/results/rfr_results.csv")
print("    reports/figures/rfr_scatter.png")
print("    reports/figures/rfr_feature_importance.png")
print("    reports/figures/rfr_n_estimators_curve.png")

print_section("DONE — RFR.py hoàn thành!")
