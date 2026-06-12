"""
=======================================================================
  K-MEANS CLUSTERING — Wine Quality Dataset
=======================================================================
  Mục tiêu: Tìm ra các cụm (cluster) tự nhiên trong dữ liệu rượu,
  sau đó phân tích xem các cụm này có khớp với nhãn chất lượng thực
  tế không (low/medium/high).

  Data:
    clustering/kmeans_full_scaled.csv
    (5318 mẫu, đã StandardScaled — toàn bộ tập dữ liệu)

  Phân tích gồm:
    1. Elbow Method  — tìm k tối ưu qua Inertia
    2. Silhouette    — xác nhận k tối ưu qua hệ số Silhouette
    3. Davies-Bouldin & Calinski-Harabasz — thêm góc nhìn đánh giá
    4. Cluster Profiling — đặc điểm trung bình của từng cụm
    5. So sánh với nhãn thực tế (low/medium/high)

  Cách chạy (từ thư mục gốc dự án):
    python notebooks/KMeans.py

  Output:
    models/kmeans_k3.pkl             — model K-means với k=3
    models/kmeans_best.pkl           — model K-means với k tối ưu
    reports/results/kmeans_metrics.csv
    reports/results/kmeans_cluster_profile.csv
    reports/figures/kmeans_*.png
=======================================================================
"""

# ── 0. Setup ──────────────────────────────────────────────────────────
import sys, os
NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, NOTEBOOK_DIR)
from utils import *

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix,
)

SCRIPT = "KMeans"
print_section(f"SCRIPT: {SCRIPT}  —  Wine Quality Dataset Clustering")
print(f"  Base dir   : {BASE}")
print(f"  Results dir: {RESULTS_DIR}")
print(f"  Figures dir: {FIG_DIR}")

# ── 0.1 Tạo thư mục cho K-means results ──────────────────────────────
KMEANS_FIG_DIR = os.path.join(FIG_DIR, "kmeans")
os.makedirs(KMEANS_FIG_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
#  PHẦN 1: LOAD DATA
# ═══════════════════════════════════════════════════════════════════════
print_section("PHẦN 1: Load Data")

X, y_quality, y_label = load_kmeans_data()
feat_names = list(X.columns)
X_arr = X.values                # numpy array cho sklearn

print(f"\n  Tổng số mẫu   : {X_arr.shape[0]}")
print(f"  Số features   : {X_arr.shape[1]}")
print(f"  Features      : {feat_names}")
print(f"\n  Phân bố quality score: {dict(zip(*np.unique(y_quality, return_counts=True)))}")
print(f"  Phân bố nhãn 3-class :")
for i, lbl in enumerate(LABEL_NAMES):
    cnt = int((y_label == i).sum())
    print(f"    {i} ({lbl:>6}) : {cnt} ({cnt/len(y_label)*100:.1f}%)")


# ═══════════════════════════════════════════════════════════════════════
#  PHẦN 2: ELBOW METHOD — TÌM K TỐI ƯU
# ═══════════════════════════════════════════════════════════════════════
print_section("PHẦN 2: Elbow Method (Inertia vs k)")

K_RANGE = range(2, 12)           # Thử k từ 2 đến 11
inertias        = []             # Within-cluster sum of squares
sil_scores      = []             # Silhouette score
db_scores       = []             # Davies-Bouldin score (nhỏ hơn = tốt hơn)
ch_scores       = []             # Calinski-Harabasz score (lớn hơn = tốt hơn)

print("\n  k | Inertia       | Silhouette | Davies-Bouldin | Calinski-Harabasz")
print("  " + "-" * 70)

for k in K_RANGE:
    km = KMeans(
        n_clusters=k,
        init="k-means++",    # Khởi tạo thông minh (tránh local minima)
        n_init=10,           # Chạy 10 lần với seed khác nhau, lấy lần tốt nhất
        max_iter=300,
        random_state=42,
        algorithm="lloyd"
    )
    labels_k = km.fit_predict(X_arr)

    ine  = km.inertia_
    sil  = silhouette_score(X_arr, labels_k, sample_size=2000, random_state=42)
    db   = davies_bouldin_score(X_arr, labels_k)
    ch   = calinski_harabasz_score(X_arr, labels_k)

    inertias.append(ine)
    sil_scores.append(sil)
    db_scores.append(db)
    ch_scores.append(ch)

    print(f"  {k:>2} | {ine:>13.1f} | {sil:>10.4f} | {db:>14.4f} | {ch:>17.1f}")

# ── 2.1 Vẽ 4-panel selection chart ───────────────────────────────────
fig_k, axes_k = plt.subplots(2, 2, figsize=(13, 9))
fig_k.suptitle("K-Means — Lựa chọn số cụm k tối ưu", fontsize=14, fontweight="bold")

ks = list(K_RANGE)

# Elbow (Inertia)
axes_k[0, 0].plot(ks, inertias, "o-", color="#5b8ec4", linewidth=2, markersize=7)
axes_k[0, 0].set_xlabel("Số cụm k"); axes_k[0, 0].set_ylabel("Inertia (WCSS)")
axes_k[0, 0].set_title("Elbow Method — Inertia")
axes_k[0, 0].set_xticks(ks)

# Silhouette
axes_k[0, 1].plot(ks, sil_scores, "s-", color="#e07b54", linewidth=2, markersize=7)
axes_k[0, 1].set_xlabel("Số cụm k"); axes_k[0, 1].set_ylabel("Silhouette Score")
axes_k[0, 1].set_title("Silhouette Score (cao = tốt)")
axes_k[0, 1].set_xticks(ks)

# Davies-Bouldin
axes_k[1, 0].plot(ks, db_scores, "^-", color="#6ec47a", linewidth=2, markersize=7)
axes_k[1, 0].set_xlabel("Số cụm k"); axes_k[1, 0].set_ylabel("Davies-Bouldin Score")
axes_k[1, 0].set_title("Davies-Bouldin (thấp = tốt)")
axes_k[1, 0].set_xticks(ks)

# Calinski-Harabasz
axes_k[1, 1].plot(ks, ch_scores, "D-", color="#c47a5b", linewidth=2, markersize=7)
axes_k[1, 1].set_xlabel("Số cụm k"); axes_k[1, 1].set_ylabel("Calinski-Harabasz Score")
axes_k[1, 1].set_title("Calinski-Harabasz (cao = tốt)")
axes_k[1, 1].set_xticks(ks)

plt.tight_layout()
fig_k.savefig(os.path.join(KMEANS_FIG_DIR, "kmeans_k_selection.png"), dpi=130, bbox_inches="tight")
plt.close()
print(f"\n  Plot saved: kmeans/kmeans_k_selection.png")

# ── 2.2 Xác định k tối ưu tự động ────────────────────────────────────
# Theo nhiều tiêu chí: Silhouette cao nhất + đây là bài toán wine quality
# có 3 nhãn thực tế → thử cả k=3 (để so sánh) và k từ silhouette tốt nhất
k_best_sil = ks[int(np.argmax(sil_scores))]
k_best_db  = ks[int(np.argmin(db_scores))]
k_best_ch  = ks[int(np.argmax(ch_scores))]

print(f"\n  k tốt nhất theo Silhouette     : k={k_best_sil}  (score={max(sil_scores):.4f})")
print(f"  k tốt nhất theo Davies-Bouldin  : k={k_best_db}   (score={min(db_scores):.4f})")
print(f"  k tốt nhất theo Calinski-Harab. : k={k_best_ch}   (score={max(ch_scores):.1f})")
print(f"\n  → Sẽ phân tích chi tiết với cả k=3 (tương ứng 3 nhãn chất lượng)")
print(f"    và k={k_best_sil} (k tốt nhất theo Silhouette)")


# ═══════════════════════════════════════════════════════════════════════
#  PHẦN 3: PHÂN TÍCH CHI TIẾT — K=3 (tương ứng nhãn low/med/high)
# ═══════════════════════════════════════════════════════════════════════
print_section("PHẦN 3: Phân tích chi tiết K=3 (so sánh với nhãn thực tế)")

km3 = KMeans(n_clusters=3, init="k-means++", n_init=20,
             max_iter=500, random_state=42, algorithm="lloyd")
labels3 = km3.fit_predict(X_arr)

sil3  = silhouette_score(X_arr, labels3)
db3   = davies_bouldin_score(X_arr, labels3)
ch3   = calinski_harabasz_score(X_arr, labels3)
ari3  = adjusted_rand_score(y_label, labels3)         # So sánh với nhãn thực
nmi3  = normalized_mutual_info_score(y_label, labels3)

print(f"\n  Inertia          : {km3.inertia_:.2f}")
print(f"  Silhouette Score : {sil3:.4f}")
print(f"  Davies-Bouldin   : {db3:.4f}")
print(f"  Calinski-Harabasz: {ch3:.1f}")
print(f"  Adjusted Rand Index (vs y_label)          : {ari3:.4f}  (1=hoàn hảo, 0=ngẫu nhiên)")
print(f"  Normalized Mutual Info (vs y_label)       : {nmi3:.4f}")

# ── 3.1 Cluster Profiling — đặc điểm trung bình của từng cụm ────────
print_section("3.1 Cluster Profiling (k=3)")

df_clust = X.copy()
df_clust["cluster"]       = labels3
df_clust["quality"]       = y_quality
df_clust["quality_label"] = y_label

profile = df_clust.groupby("cluster").agg(
    n_samples=("quality", "count"),
    avg_quality=("quality", "mean"),
    std_quality=("quality", "std"),
    **{f"avg_{f}": (f, "mean") for f in feat_names}
).round(4)
print(f"\n  Cluster profile (k=3):\n")
print(profile.to_string())

# Lưu profile
profile_path = os.path.join(RESULTS_DIR, "kmeans_cluster_profile_k3.csv")
profile.to_csv(profile_path)
print(f"\n  Profile saved: {profile_path}")

# ── 3.2 Contingency Table (Cụm vs Nhãn thực) ─────────────────────────
print_section("3.2 Contingency Table — Cluster vs Nhãn thực (k=3)")
cont = pd.crosstab(
    pd.Series(labels3, name="Cluster"),
    pd.Series([LABEL_NAMES[l] for l in y_label], name="True Label")
)
print(f"\n{cont.to_string()}")

# ── 3.3 Plots k=3 ─────────────────────────────────────────────────────
print_section("3.3 Plots k=3")

# a) Confusion-style heatmap: Cluster vs True Label
fig_ct, ax_ct = plt.subplots(figsize=(7, 5))
sns.heatmap(cont, annot=True, fmt="d", cmap="YlOrRd", ax=ax_ct)
ax_ct.set_title("K-Means (k=3) — Cluster vs True Label", fontsize=12, fontweight="bold")
ax_ct.set_xlabel("True Label"); ax_ct.set_ylabel("Cluster")
fig_ct.tight_layout()
fig_ct.savefig(os.path.join(KMEANS_FIG_DIR, "kmeans_k3_contingency.png"), dpi=130, bbox_inches="tight")
plt.close()
print(f"  Plot saved: kmeans/kmeans_k3_contingency.png")

# b) Box plot — phân phối quality score theo cluster
fig_bx, ax_bx = plt.subplots(figsize=(8, 5))
data_bp = [y_quality[labels3 == c] for c in range(3)]
bp = ax_bx.boxplot(data_bp, labels=[f"Cluster {i}" for i in range(3)],
                    patch_artist=True, medianprops=dict(color="white", linewidth=2))
colors = ["#5b8ec4", "#e07b54", "#6ec47a"]
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c)
ax_bx.set_ylabel("Quality Score (3-9)")
ax_bx.set_title("K-Means (k=3) — Phân phối Quality Score theo Cluster", fontsize=12, fontweight="bold")
fig_bx.tight_layout()
fig_bx.savefig(os.path.join(KMEANS_FIG_DIR, "kmeans_k3_quality_boxplot.png"), dpi=130, bbox_inches="tight")
plt.close()
print(f"  Plot saved: kmeans/kmeans_k3_quality_boxplot.png")

# c) Radar chart — Feature profile của từng cluster (Top 8 features)
top8 = feat_names[:8]  # Lấy 8 features đầu
centers_norm = pd.DataFrame(km3.cluster_centers_, columns=feat_names)[top8]
n_feat = len(top8)
angles = [n / float(n_feat) * 2 * np.pi for n in range(n_feat)]
angles += angles[:1]   # Đóng vòng

fig_rad, ax_rad = plt.subplots(figsize=(8, 7), subplot_kw=dict(polar=True))
colors_rad = ["#5b8ec4", "#e07b54", "#6ec47a"]
for i in range(3):
    vals = centers_norm.iloc[i].tolist()
    vals += vals[:1]
    ax_rad.plot(angles, vals, "o-", linewidth=2, label=f"Cluster {i}", color=colors_rad[i])
    ax_rad.fill(angles, vals, alpha=0.15, color=colors_rad[i])
ax_rad.set_xticks(angles[:-1])
ax_rad.set_xticklabels(top8, size=8)
ax_rad.set_title("K-Means (k=3) — Cluster Centers Radar Chart\n(Scaled feature means)", 
                  fontsize=11, fontweight="bold", pad=15)
ax_rad.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
fig_rad.tight_layout()
fig_rad.savefig(os.path.join(KMEANS_FIG_DIR, "kmeans_k3_radar.png"), dpi=130, bbox_inches="tight")
plt.close()
print(f"  Plot saved: kmeans/kmeans_k3_radar.png")

# d) Silhouette Plot cho k=3
sil_vals = silhouette_samples(X_arr, labels3)
fig_sil, ax_sil = plt.subplots(figsize=(9, 6))
y_lower = 10
for i in range(3):
    sil_i = np.sort(sil_vals[labels3 == i])
    size_i = sil_i.shape[0]
    y_upper = y_lower + size_i
    ax_sil.fill_betweenx(np.arange(y_lower, y_upper), 0, sil_i,
                          facecolor=colors[i], edgecolor="none", alpha=0.8)
    ax_sil.text(-0.05, y_lower + 0.5 * size_i, str(i))
    y_lower = y_upper + 10

ax_sil.axvline(x=sil3, color="red", linestyle="--", label=f"Mean={sil3:.4f}")
ax_sil.set_xlabel("Silhouette Coefficient")
ax_sil.set_title(f"Silhouette Plot — K-Means k=3 (mean={sil3:.4f})", fontsize=12, fontweight="bold")
ax_sil.legend()
fig_sil.tight_layout()
fig_sil.savefig(os.path.join(KMEANS_FIG_DIR, "kmeans_k3_silhouette.png"), dpi=130, bbox_inches="tight")
plt.close()
print(f"  Plot saved: kmeans/kmeans_k3_silhouette.png")


# ═══════════════════════════════════════════════════════════════════════
#  PHẦN 4: PHÂN TÍCH CHI TIẾT — K=k_best_sil
# ═══════════════════════════════════════════════════════════════════════
if k_best_sil != 3:
    print_section(f"PHẦN 4: Phân tích chi tiết K={k_best_sil} (Silhouette tốt nhất)")

    km_best = KMeans(n_clusters=k_best_sil, init="k-means++", n_init=20,
                     max_iter=500, random_state=42, algorithm="lloyd")
    labels_best = km_best.fit_predict(X_arr)

    sil_b = silhouette_score(X_arr, labels_best)
    db_b  = davies_bouldin_score(X_arr, labels_best)

    print(f"\n  k={k_best_sil} | Silhouette={sil_b:.4f} | Davies-Bouldin={db_b:.4f}")

    # Box plot
    fig_bx2, ax_bx2 = plt.subplots(figsize=(max(8, k_best_sil * 1.5), 5))
    data_bp2 = [y_quality[labels_best == c] for c in range(k_best_sil)]
    bp2 = ax_bx2.boxplot(data_bp2, labels=[f"Cluster {i}" for i in range(k_best_sil)],
                          patch_artist=True, medianprops=dict(color="white", linewidth=2))
    cmap2 = plt.cm.tab10
    for idx, patch in enumerate(bp2["boxes"]):
        patch.set_facecolor(cmap2(idx / k_best_sil))
    ax_bx2.set_ylabel("Quality Score (3-9)")
    ax_bx2.set_title(f"K-Means (k={k_best_sil}) — Phân phối Quality Score", fontsize=12, fontweight="bold")
    fig_bx2.tight_layout()
    fig_bx2.savefig(os.path.join(KMEANS_FIG_DIR, f"kmeans_k{k_best_sil}_quality_boxplot.png"),
                    dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: kmeans/kmeans_k{k_best_sil}_quality_boxplot.png")

    save_model(km_best, f"kmeans_k{k_best_sil}_best.pkl")
    K_BEST = k_best_sil
    KM_BEST = km_best
    LABELS_BEST = labels_best
else:
    K_BEST = 3
    KM_BEST = km3
    LABELS_BEST = labels3
    print_section(f"PHẦN 4: k_best_sil = k=3 — Bỏ qua phân tích trùng lặp")

# ── Lưu model k=3 và k=best ────────────────────────────────────────────
save_model(km3, "kmeans_k3.pkl")
if K_BEST != 3:
    save_model(KM_BEST, "kmeans_best.pkl")
else:
    save_model(km3, "kmeans_best.pkl")    # cùng model


# ═══════════════════════════════════════════════════════════════════════
#  PHẦN 5: LƯU KẾT QUẢ TẤT CẢ K
# ═══════════════════════════════════════════════════════════════════════
print_section("PHẦN 5: Lưu tổng hợp metrics tất cả k")

# ARI và NMI cho tất cả k
all_metrics = []
for i, k in enumerate(K_RANGE):
    km_tmp = KMeans(n_clusters=k, init="k-means++", n_init=10,
                    max_iter=300, random_state=42)
    lbl_tmp = km_tmp.fit_predict(X_arr)
    ari_tmp = adjusted_rand_score(y_label, lbl_tmp)
    nmi_tmp = normalized_mutual_info_score(y_label, lbl_tmp)
    all_metrics.append({
        "k": k,
        "inertia": round(inertias[i], 2),
        "silhouette": round(sil_scores[i], 4),
        "davies_bouldin": round(db_scores[i], 4),
        "calinski_harabasz": round(ch_scores[i], 1),
        "adjusted_rand_index_vs_3label": round(ari_tmp, 4),
        "normalized_mutual_info": round(nmi_tmp, 4),
    })

metrics_df = pd.DataFrame(all_metrics)
metrics_path = os.path.join(RESULTS_DIR, "kmeans_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)
print(f"\n  Metrics saved: {metrics_path}")
print(f"\n{metrics_df.to_string(index=False)}")


# ═══════════════════════════════════════════════════════════════════════
#  TỔNG KẾT
# ═══════════════════════════════════════════════════════════════════════
print_section("TỔNG KẾT — K-Means Clustering")

print(f"""
  Kết quả chính:
  ┌─────────────────────────────────────────────────────────┐
  │  k tối ưu (Silhouette)     : k = {k_best_sil:<4}                   │
  │  k theo nhãn thực tế       : k = 3                     │
  │                                                         │
  │  K=3 Metrics:                                           │
  │    Inertia            : {km3.inertia_:>14.2f}             │
  │    Silhouette Score   : {sil3:>14.4f}             │
  │    Davies-Bouldin     : {db3:>14.4f}             │
  │    Calinski-Harabasz  : {ch3:>14.1f}             │
  │    Adjusted Rand Index: {ari3:>14.4f}             │
  │    Normalized Mut.Info: {nmi3:>14.4f}             │
  └─────────────────────────────────────────────────────────┘

  Diễn giải ARI:
    0.0  → Phân cụm ngẫu nhiên, không liên quan nhãn thực
    0.2  → Yếu
    0.4  → Trung bình
    0.6  → Tốt
    1.0  → Hoàn hảo

  Files đã lưu:
    models/kmeans_k3.pkl
    models/kmeans_best.pkl
    reports/results/kmeans_metrics.csv
    reports/results/kmeans_cluster_profile_k3.csv
    reports/figures/kmeans/kmeans_k_selection.png
    reports/figures/kmeans/kmeans_k3_*.png
""")

print_section("DONE — KMeans.py hoàn thành!")
