"""
=======================================================================
  Wine Quality Dataset -- EDA & Data Preprocessing
=======================================================================
  Dataset : UCI Wine Quality (Red + White)
  Target  : EDA + prepare data for ML modeling
=======================================================================
"""

# ── 0. Imports ───────────────────────────────────────────────────────
import os, sys, io, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(BASE, "data", "raw")
PROC_DIR = os.path.join(BASE, "data", "processed")
FIG_DIR  = os.path.join(BASE, "reports", "figures")
for d in [PROC_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Helper: robust CSV reader for this dataset ────────────────────────
def read_wine_csv(path):
    """
    The wine CSV header is wrapped in outer quotes with "" inner escapes.
    This helper parses it correctly regardless of platform/encoding.
    """
    with open(path, "r") as f:
        lines = f.readlines()
    hdr = lines[0].strip()
    if hdr.startswith('"') and hdr.endswith('"'):
        hdr = hdr[1:-1]
    cols = [c.replace('""', '').strip('"') for c in hdr.split(';')]
    df = pd.read_csv(io.StringIO("".join(lines[1:])), sep=";", header=None, names=cols)
    return df

# ═══════════════════════════════════════════════════════════════════════
#  STEP 1: LOAD DATA
# ═══════════════════════════════════════════════════════════════════════
sep = "=" * 65
print(sep); print("  STEP 1: LOAD DATA"); print(sep)

df_red   = read_wine_csv(os.path.join(RAW_DIR, "winequality-red.csv"))
df_white = read_wine_csv(os.path.join(RAW_DIR, "winequality-white.csv"))
df_red  ["wine_type"] = "red"
df_white["wine_type"] = "white"

df = pd.concat([df_red, df_white], ignore_index=True)
FEATURES = [c for c in df.columns if c not in ("quality", "wine_type")]
TARGET   = "quality"

print(f"  Red wine   : {len(df_red):>5} samples")
print(f"  White wine : {len(df_white):>5} samples")
print(f"  Combined   : {len(df):>5} samples,  {df.shape[1]} columns")
print(f"  Features   : {FEATURES}")

# ═══════════════════════════════════════════════════════════════════════
#  STEP 2: BASIC INSPECTION & DESCRIPTIVE STATISTICS
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{sep}\n  STEP 2: BASIC INSPECTION\n{sep}")

print("\n[2.1] Data types")
print(df.dtypes)

print("\n[2.2] Missing values")
mv = df.isnull().sum()
mv_df = pd.DataFrame({"count": mv, "pct (%)": (mv/len(df)*100).round(2)})
mv_df = mv_df[mv_df["count"] > 0]
print("  None!" if mv_df.empty else mv_df.to_string())

print("\n[2.3] Descriptive statistics")
desc = df[FEATURES + [TARGET]].describe().T
desc["CV (%)"] = (desc["std"] / desc["mean"] * 100).round(2)
print(desc.to_string())

# ═══════════════════════════════════════════════════════════════════════
#  STEP 3: TARGET VARIABLE
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{sep}\n  STEP 3: TARGET VARIABLE — quality\n{sep}")

print("\n  Distribution (combined):")
print(df[TARGET].value_counts().sort_index().to_string())
print(f"\n  Median={df[TARGET].median()}, Mean={df[TARGET].mean():.3f}, Std={df[TARGET].std():.3f}")
print("\n  Cross-table wine_type x quality:")
print(pd.crosstab(df["wine_type"], df[TARGET]).to_string())

# ═══════════════════════════════════════════════════════════════════════
#  STEP 4: OUTLIER ANALYSIS (IQR)
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{sep}\n  STEP 4: OUTLIER ANALYSIS (IQR method)\n{sep}")

rows = []
for col in FEATURES:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
    rows.append({"Feature": col, "Q1": Q1, "Q3": Q3, "IQR": IQR,
                 "Lower": lo, "Upper": hi,
                 "Outliers": n_out, "Pct (%)": round(n_out/len(df)*100, 2)})
out_df = pd.DataFrame(rows).set_index("Feature").round(4)
print(out_df.to_string())

# ═══════════════════════════════════════════════════════════════════════
#  STEP 5: CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{sep}\n  STEP 5: CORRELATION ANALYSIS\n{sep}")

corr = df[FEATURES + [TARGET]].corr()

upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
strong = (upper.stack()
               .reset_index()
               .rename(columns={"level_0":"Var1","level_1":"Var2",0:"r"})
               .assign(abs_r=lambda x: x["r"].abs())
               .query("abs_r > 0.5")
               .sort_values("abs_r", ascending=False))
print("\n  Pairs with |r| > 0.5:")
print(strong.to_string(index=False))

tgt_corr = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
print("\n  Correlation with 'quality':")
print(tgt_corr.round(4).to_string())

# ═══════════════════════════════════════════════════════════════════════
#  STEP 6: DUPLICATE CHECK
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{sep}\n  STEP 6: DUPLICATE CHECK\n{sep}")
dup = df.drop(columns="wine_type").duplicated().sum()
print(f"  Duplicate rows: {dup} ({dup/len(df)*100:.2f}%)")

# ═══════════════════════════════════════════════════════════════════════
#  STEP 7: VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{sep}\n  STEP 7: VISUALIZATION\n{sep}")

sns.set_theme(style="darkgrid", palette="muted", font_scale=1.0)
PAL = {"red": "#e05252", "white": "#6699cc"}

# -- Fig 1: Quality distribution
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Target Variable Distribution — Wine Quality", fontsize=14, fontweight="bold")
vc_all = df[TARGET].value_counts().sort_index()
axes[0].bar(vc_all.index, vc_all.values, color="#7b9cc4", edgecolor="white", width=0.7)
axes[0].set_title("Combined (Red + White)"); axes[0].set_xlabel("Quality"); axes[0].set_ylabel("Count")
for x, y in zip(vc_all.index, vc_all.values):
    axes[0].text(x, y+10, str(y), ha="center", fontsize=8)

for ax, wt, c in zip(axes[1:], ["red","white"], ["#e05252","#6699cc"]):
    vc = df[df["wine_type"]==wt][TARGET].value_counts().sort_index()
    ax.bar(vc.index, vc.values, color=c, edgecolor="white", width=0.7)
    ax.set_title(f"{wt.capitalize()} Wine"); ax.set_xlabel("Quality"); ax.set_ylabel("Count")
    for x, y in zip(vc.index, vc.values):
        ax.text(x, y+3, str(y), ha="center", fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig01_quality_distribution.png"), dpi=130, bbox_inches="tight")
plt.close()
print("  [OK] fig01_quality_distribution.png")

# -- Fig 2: Feature histograms (drop any NaN columns)
valid_feats = [c for c in FEATURES if not df[c].isnull().all()]
n_cols, n_rows = 4, int(np.ceil(len(valid_feats)/4))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5*n_rows))
fig.suptitle("Feature Distributions (Red vs White Wine)", fontsize=14, fontweight="bold")
axes = axes.flatten()
for i, col in enumerate(valid_feats):
    for wt, c in PAL.items():
        data = df[df["wine_type"]==wt][col].dropna()
        axes[i].hist(data, bins=30, alpha=0.55, color=c, label=wt.capitalize(), edgecolor="none")
    axes[i].set_title(col, fontsize=9); axes[i].tick_params(labelsize=8)
    if i == 0: axes[i].legend(fontsize=8)
for ax in axes[len(valid_feats):]: ax.set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig02_feature_distributions.png"), dpi=130, bbox_inches="tight")
plt.close()
print("  [OK] fig02_feature_distributions.png")

# -- Fig 3: Boxplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5*n_rows))
fig.suptitle("Boxplots — Outlier Detection per Feature", fontsize=14, fontweight="bold")
axes = axes.flatten()
for i, col in enumerate(valid_feats):
    data = [df[df["wine_type"]==wt][col].dropna().values for wt in ["red","white"]]
    bp = axes[i].boxplot(data, labels=["Red","White"], patch_artist=True,
                         medianprops=dict(color="white", linewidth=2))
    bp["boxes"][0].set_facecolor("#e05252")
    bp["boxes"][1].set_facecolor("#6699cc")
    axes[i].set_title(col, fontsize=9); axes[i].tick_params(labelsize=8)
for ax in axes[len(valid_feats):]: ax.set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig03_boxplots_outlier.png"), dpi=130, bbox_inches="tight")
plt.close()
print("  [OK] fig03_boxplots_outlier.png")

# -- Fig 4: Correlation heatmap
fig, ax = plt.subplots(figsize=(13, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(230, 20, as_cmap=True),
            vmax=1, vmin=-1, center=0, square=True, linewidths=0.5,
            annot=True, fmt=".2f", annot_kws={"size": 8}, ax=ax)
ax.set_title("Pearson Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig04_correlation_heatmap.png"), dpi=130, bbox_inches="tight")
plt.close()
print("  [OK] fig04_correlation_heatmap.png")

# -- Fig 5: Scatter — top 6 features vs quality
top6 = tgt_corr.abs().nlargest(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Scatter — Top 6 Features correlated with Quality", fontsize=13, fontweight="bold")
axes = axes.flatten()
for i, col in enumerate(top6):
    for wt, c in PAL.items():
        sub = df[df["wine_type"]==wt]
        axes[i].scatter(sub[col], sub[TARGET], alpha=0.18, s=12, c=c, label=wt.capitalize())
    r = df[col].corr(df[TARGET])
    axes[i].set_xlabel(col, fontsize=9); axes[i].set_ylabel("Quality", fontsize=9)
    axes[i].set_title(f"{col}  (r={r:.3f})", fontsize=10)
    if i == 0: axes[i].legend(fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig05_scatter_top_features.png"), dpi=130, bbox_inches="tight")
plt.close()
print("  [OK] fig05_scatter_top_features.png")

# -- Fig 6: Violin plots
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Violin Plot — Feature Distribution across Quality Groups", fontsize=13, fontweight="bold")
axes = axes.flatten()
for i, col in enumerate(top6):
    sns.violinplot(data=df, x=TARGET, y=col, hue="wine_type",
                   palette=PAL, split=False, inner="quartile",
                   ax=axes[i], legend=(i==0))
    axes[i].set_title(col, fontsize=10); axes[i].set_xlabel("Quality", fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig06_violin_quality.png"), dpi=130, bbox_inches="tight")
plt.close()
print("  [OK] fig06_violin_quality.png")

# -- Fig 7: Mean quality by quantile (feature monotonicity)
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Mean Quality by Feature Quantile (Monotonicity Check)", fontsize=13, fontweight="bold")
axes = axes.flatten()
for i, col in enumerate(top6):
    df["_q"] = pd.qcut(df[col], q=5, duplicates="drop")
    grp = df.groupby("_q", observed=True)[TARGET].mean()
    axes[i].bar(range(len(grp)), grp.values, color="#7b9cc4", edgecolor="white")
    axes[i].set_xticks(range(len(grp)))
    axes[i].set_xticklabels([str(v)[:14] for v in grp.index], rotation=25, fontsize=7)
    axes[i].set_title(col, fontsize=10); axes[i].set_ylabel("Avg Quality")
df.drop(columns="_q", inplace=True)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig07_mean_quality_by_quantile.png"), dpi=130, bbox_inches="tight")
plt.close()
print("  [OK] fig07_mean_quality_by_quantile.png")

# -- Fig 8: Pairplot (alcohol, volatile acidity, density, quality)
pair_cols = ["alcohol", "volatile acidity", "density", "chlorides"]
pp_data = df[pair_cols + [TARGET, "wine_type"]].copy()
pp_data[TARGET] = pp_data[TARGET].astype(str)
try:
    g = sns.pairplot(pp_data, hue="wine_type", palette=PAL, diag_kind="kde",
                     plot_kws={"alpha": 0.25, "s": 15})
    g.figure.suptitle("Pairplot — Key Features", y=1.01, fontsize=13, fontweight="bold")
    g.figure.savefig(os.path.join(FIG_DIR, "fig08_pairplot.png"), dpi=110, bbox_inches="tight")
    plt.close()
    print("  [OK] fig08_pairplot.png")
except Exception as e:
    print(f"  [SKIP] fig08_pairplot: {e}")

# ═══════════════════════════════════════════════════════════════════════
#  STEP 8: DATA PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{sep}\n  STEP 8: DATA PREPROCESSING\n{sep}")

# 8.1 Remove duplicates
df_clean = df.copy()
before = len(df_clean)
df_clean = df_clean.drop_duplicates(subset=FEATURES+[TARGET]).reset_index(drop=True)
after = len(df_clean)
print(f"\n[8.1] Duplicate removal: {before} -> {after}  (removed {before-after} rows)")

# 8.2 Winsorization (clip at 1%-99% percentile, per wine_type group for fairness)
df_win = df_clean.copy()
for col in FEATURES:
    lo, hi = df_win[col].quantile(0.01), df_win[col].quantile(0.99)
    df_win[col] = df_win[col].clip(lo, hi)
print(f"[8.2] Winsorization (1%-99%) applied to {len(FEATURES)} features  -> shape {df_win.shape}")

# Verify no NaN after winsorization
assert df_win[FEATURES].isnull().sum().sum() == 0, "NaN found after winsorization!"

# 8.3 Feature Engineering
df_fe = df_win.copy()
df_fe["free_total_SO2_ratio"] = df_fe["free sulfur dioxide"] / (df_fe["total sulfur dioxide"] + 1e-6)
df_fe["acid_ratio"]           = df_fe["fixed acidity"]        / (df_fe["volatile acidity"]     + 1e-6)
df_fe["sulphate_alcohol"]     = df_fe["sulphates"]            * df_fe["alcohol"]
df_fe["is_red"]               = (df_fe["wine_type"] == "red").astype(int)
df_fe["quality_label"]        = pd.cut(df_fe[TARGET], bins=[0,4,6,10], labels=["low","medium","high"])

NEW_FEATS = ["free_total_SO2_ratio","acid_ratio","sulphate_alcohol","is_red"]
ALL_FEATS = FEATURES + NEW_FEATS

print(f"[8.3] New features: {NEW_FEATS}")
print("      Quality label distribution:")
print(pd.crosstab(df_fe["wine_type"], df_fe["quality_label"]).to_string())

# 8.4 Scaling
X = df_fe[ALL_FEATS].copy()
y = df_fe[TARGET].copy()

scaler_std  = StandardScaler(); X_std    = pd.DataFrame(scaler_std.fit_transform(X),  columns=ALL_FEATS)
scaler_mm   = MinMaxScaler();   X_minmax = pd.DataFrame(scaler_mm.fit_transform(X),   columns=ALL_FEATS)
scaler_rob  = RobustScaler();   X_robust = pd.DataFrame(scaler_rob.fit_transform(X),  columns=ALL_FEATS)
print(f"[8.4] Three scalers applied: StandardScaler | MinMaxScaler | RobustScaler")

# Scaling comparison figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Scaling Comparison — Feature Ranges After Transformation", fontsize=13, fontweight="bold")
labels = ["StandardScaler\n(mean=0, std=1)", "MinMaxScaler\n([0,1])", "RobustScaler\n(median-based)"]
for ax, Xs, lbl in zip(axes, [X_std, X_minmax, X_robust], labels):
    ax.boxplot([Xs[c].values for c in ALL_FEATS], labels=ALL_FEATS, vert=False)
    ax.set_title(lbl, fontsize=10); ax.tick_params(labelsize=7)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig09_scaling_comparison.png"), dpi=130, bbox_inches="tight")
plt.close()
print("  [OK] fig09_scaling_comparison.png")

# 8.5 Train / Val / Test split (70 / 15 / 15, stratified by quality)
X_tv, X_test, y_tv, y_test  = train_test_split(X_std, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.176, random_state=42, stratify=y_tv)

pct_tr = len(X_train)/len(X)*100
pct_vl = len(X_val)/len(X)*100
pct_ts = len(X_test)/len(X)*100
print(f"[8.5] Stratified split:")
print(f"      Train : {len(X_train):>5} ({pct_tr:.1f}%)")
print(f"      Val   : {len(X_val):>5}   ({pct_vl:.1f}%)")
print(f"      Test  : {len(X_test):>5}  ({pct_ts:.1f}%)")

# Verify stratification
print("\n      Quality distribution per split:")
for name, yy in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    dist = yy.value_counts(normalize=True).sort_index().round(3).to_dict()
    print(f"      {name}: {dist}")

# ═══════════════════════════════════════════════════════════════════════
#  STEP 9: SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{sep}\n  STEP 9: SAVE PROCESSED DATA\n{sep}")

# Full processed dataset (no scaling, with FE columns)
df_final = df_fe[ALL_FEATS + [TARGET, "quality_label", "wine_type"]].copy()
df_final.to_csv(os.path.join(PROC_DIR, "wine_combined_processed.csv"), index=False)
print(f"  Saved: wine_combined_processed.csv  {df_final.shape}")

# Train / Val / Test (StandardScaled)
for name, Xp, yp in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
    out = Xp.copy(); out[TARGET] = yp.values
    out.to_csv(os.path.join(PROC_DIR, f"{name}.csv"), index=False)
    print(f"  Saved: {name}.csv  {out.shape}")

# MinMax & Robust
X_minmax.assign(**{TARGET: y.values}).to_csv(os.path.join(PROC_DIR, "wine_minmax_scaled.csv"), index=False)
X_robust.assign(**{TARGET: y.values}).to_csv(os.path.join(PROC_DIR, "wine_robust_scaled.csv"), index=False)
print(f"  Saved: wine_minmax_scaled.csv  {X_minmax.shape}")
print(f"  Saved: wine_robust_scaled.csv  {X_robust.shape}")

# ═══════════════════════════════════════════════════════════════════════
#  STEP 10: SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{sep}")
print("  STEP 10: SUMMARY")
print(sep)
print(f"""
  DATASET
    Red wine   : 1,599 samples
    White wine : 4,898 samples
    Combined   : 6,497 -> {len(df_final)} samples (after dedup)
    Missing values : None
    Duplicates removed : {before - after} rows ({(before-after)/before*100:.1f}%)

  EDA KEY FINDINGS
    Quality range : {int(df[TARGET].min())} - {int(df[TARGET].max())} (concentrated at 5-7)
    Top feature correlated with quality : {tgt_corr.index[0]} (r={tgt_corr.iloc[0]:.3f})
    High multicollinearity pairs:
{chr(10).join("      "+r["Var1"]+" <-> "+r["Var2"]+f" (r={r['r']:.3f})" for _,r in strong.iterrows())}

  PREPROCESSING STEPS
    [1] Remove duplicates
    [2] Winsorization at [1%, 99%] per feature
    [3] Feature Engineering: free_total_SO2_ratio, acid_ratio, sulphate_alcohol, is_red
    [4] Scaling: StandardScaler / MinMaxScaler / RobustScaler (all saved)
    [5] Stratified split: {pct_tr:.0f}% Train | {pct_vl:.0f}% Val | {pct_ts:.0f}% Test

  OUTPUT
    data/processed/wine_combined_processed.csv  (FE done, no scaling)
    data/processed/train.csv / val.csv / test.csv  (StandardScaled)
    data/processed/wine_minmax_scaled.csv
    data/processed/wine_robust_scaled.csv
    reports/figures/ -> 9 PNG charts
""")
print(sep)
print("  ALL DONE!")
print(sep)
