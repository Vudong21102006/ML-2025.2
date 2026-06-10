"""
02_prepare_additional_data.py  (v2 — full coverage)
=====================================================
Tạo ĐẦY ĐỦ tất cả data files cần thiết cho mọi model:

  REGRESSION (target = quality score 3-9)
  ├── train.csv / val.csv / test.csv                 StandardScaled
  ├── train_unscaled.csv / val_unscaled.csv / test_unscaled.csv   Unscaled
  └── (wine_minmax_scaled.csv & wine_robust_scaled.csv)  — đã có

  CLASSIFICATION (target = quality_label_enc: 0=low, 1=medium, 2=high)
  ├── train_clf.csv / val_clf.csv / test_clf.csv     StandardScaled
  └── train_clf_unscaled.csv / val_clf_unscaled.csv / test_clf_unscaled.csv  Unscaled

  K-MEANS (clustering analysis — cần full dataset)
  └── kmeans_full_scaled.csv                         StandardScaled, tất cả 5318 mẫu + quality + label

  VERIFICATION
  └── data_summary.csv                               Tóm tắt tất cả files
"""

import os, sys, io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE, "data", "processed")
REG_DIR  = os.path.join(PROC_DIR, "regression")
CLF_DIR  = os.path.join(PROC_DIR, "classification")
CLU_DIR  = os.path.join(PROC_DIR, "clustering")
FULL_DIR = os.path.join(PROC_DIR, "full")

for d in [REG_DIR, CLF_DIR, CLU_DIR, FULL_DIR]:
    os.makedirs(d, exist_ok=True)

print("=" * 65)
print("  FULL DATA PREPARATION FOR ALL MODELS")
print("=" * 65)

# ══════════════════════════════════════════════════════════════════════
# 1. LOAD BASE DATA
# ══════════════════════════════════════════════════════════════════════
print("\n[1] Loading base data...")
df = pd.read_csv(os.path.join(PROC_DIR, "wine_combined_processed.csv"))
print(f"    wine_combined_processed.csv: {df.shape}")

# Feature columns (15 total: 11 original + 4 engineered)
FEAT_COLS = [c for c in df.columns
             if c not in ("quality", "quality_label", "wine_type")]
TARGET    = "quality"

# Encode quality_label: low=0, medium=1, high=2
label_map = {"low": 0, "medium": 1, "high": 2}
df["quality_label_enc"] = df["quality_label"].map(label_map)

print(f"    Features ({len(FEAT_COLS)}): {FEAT_COLS}")
print(f"    Quality range: {df[TARGET].min()} - {df[TARGET].max()}")
print(f"    Class distribution:")
for v, name in [(0,"low"),(1,"medium"),(2,"high")]:
    n = (df["quality_label_enc"] == v).sum()
    print(f"      {name} ({v}): {n} ({n/len(df)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════
# 2. RECREATE IDENTICAL SPLITS (same as 01_EDA_and_Preprocessing.py)
# ══════════════════════════════════════════════════════════════════════
print("\n[2] Creating train/val/test splits (random_state=42, stratified)...")

X    = df[FEAT_COLS].copy()
y_r  = df[TARGET].copy()
y_c  = df["quality_label_enc"].copy()

# Primary split: 85% temp, 15% test
X_tv, X_te, y_r_tv, y_r_te, y_c_tv, y_c_te = train_test_split(
    X, y_r, y_c, test_size=0.15, random_state=42, stratify=y_r
)
# Secondary split: ~70% train, ~15% val
X_tr, X_vl, y_r_tr, y_r_vl, y_c_tr, y_c_vl = train_test_split(
    X_tv, y_r_tv, y_c_tv, test_size=0.176, random_state=42, stratify=y_r_tv
)

print(f"    Train: {len(X_tr)} | Val: {len(X_vl)} | Test: {len(X_te)}")

# ══════════════════════════════════════════════════════════════════════
# 3. FIT STANDARD SCALER ON TRAIN ONLY (prevent data leakage)
# ══════════════════════════════════════════════════════════════════════
print("\n[3] Fitting StandardScaler on train set only...")
scaler = StandardScaler()
scaler.fit(X_tr)

X_tr_std = pd.DataFrame(scaler.transform(X_tr), columns=FEAT_COLS)
X_vl_std = pd.DataFrame(scaler.transform(X_vl), columns=FEAT_COLS)
X_te_std = pd.DataFrame(scaler.transform(X_te), columns=FEAT_COLS)

# Verify scaling
print(f"    Train mean (should be ~0): {X_tr_std.mean().abs().max():.6f}")
print(f"    Train std  (should be ~1): {X_tr_std.std().mean():.6f}")
print(f"    Val  mean  (leakage check): {X_vl_std.mean().abs().max():.6f} (small OK)")

# ══════════════════════════════════════════════════════════════════════
# 4. SAVE ALL FILE VARIANTS
# ══════════════════════════════════════════════════════════════════════
created_files = []

def save_csv(X, y_series, target_col, filename, dest_dir, reset_idx=True):
    out = X.copy()
    if reset_idx:
        out = out.reset_index(drop=True)
    out[target_col] = y_series.values
    path = os.path.join(dest_dir, filename)
    out.to_csv(path, index=False)
    size_kb = os.path.getsize(path) / 1024
    created_files.append({"file": filename, "dir": os.path.basename(dest_dir),
                          "rows": len(out), "cols": len(out.columns), "size_KB": round(size_kb,1)})
    print(f"    Saved: {os.path.basename(dest_dir)}/{filename:<40} {out.shape}  {size_kb:.0f} KB")
    return out

print("\n[4a] REGRESSION — StandardScaled")
save_csv(X_tr_std, y_r_tr, TARGET, "train.csv",          REG_DIR)
save_csv(X_vl_std, y_r_vl, TARGET, "val.csv",            REG_DIR)
save_csv(X_te_std, y_r_te, TARGET, "test.csv",           REG_DIR)

print("\n[4b] REGRESSION — Unscaled (for tree-based models)")
save_csv(X_tr, y_r_tr, TARGET, "train_unscaled.csv",     REG_DIR)
save_csv(X_vl, y_r_vl, TARGET, "val_unscaled.csv",       REG_DIR)
save_csv(X_te, y_r_te, TARGET, "test_unscaled.csv",      REG_DIR)

print("\n[4c] CLASSIFICATION — StandardScaled (3-class label)")
save_csv(X_tr_std, y_c_tr, "quality_label_enc", "train_clf.csv",          CLF_DIR)
save_csv(X_vl_std, y_c_vl, "quality_label_enc", "val_clf.csv",            CLF_DIR)
save_csv(X_te_std, y_c_te, "quality_label_enc", "test_clf.csv",           CLF_DIR)

print("\n[4d] CLASSIFICATION — Unscaled (for tree-based classifiers + LightGBM+SMOTE)")
save_csv(X_tr, y_c_tr, "quality_label_enc", "train_clf_unscaled.csv",     CLF_DIR)
save_csv(X_vl, y_c_vl, "quality_label_enc", "val_clf_unscaled.csv",       CLF_DIR)
save_csv(X_te, y_c_te, "quality_label_enc", "test_clf_unscaled.csv",      CLF_DIR)

print("\n[4e] K-MEANS — Full dataset StandardScaled (all 5318 samples)")
X_full_std = pd.DataFrame(scaler.transform(X), columns=FEAT_COLS)
X_full_std = X_full_std.reset_index(drop=True)
X_full_std[TARGET]              = y_r.values
X_full_std["quality_label_enc"] = y_c.values
path_km = os.path.join(CLU_DIR, "kmeans_full_scaled.csv")
X_full_std.to_csv(path_km, index=False)
size_kb = os.path.getsize(path_km) / 1024
created_files.append({"file": "kmeans_full_scaled.csv", "dir": "clustering",
                      "rows": len(X_full_std), "cols": len(X_full_std.columns), "size_KB": round(size_kb,1)})
print(f"    Saved: clustering/kmeans_full_scaled.csv  {X_full_std.shape}  {size_kb:.0f} KB")

# ══════════════════════════════════════════════════════════════════════
# 5. VERIFY — Kiểm tra toàn bộ files
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  [5] VERIFICATION — Checking all files")
print("=" * 65)

REQUIRED_FILES = {
    os.path.join(REG_DIR, "train.csv")                    : {"rows":3724, "target":TARGET,           "scaled":True},
    os.path.join(REG_DIR, "val.csv")                      : {"rows":796,  "target":TARGET,           "scaled":True},
    os.path.join(REG_DIR, "test.csv")                     : {"rows":798,  "target":TARGET,           "scaled":True},
    os.path.join(REG_DIR, "train_unscaled.csv")           : {"rows":3724, "target":TARGET,           "scaled":False},
    os.path.join(REG_DIR, "val_unscaled.csv")             : {"rows":796,  "target":TARGET,           "scaled":False},
    os.path.join(REG_DIR, "test_unscaled.csv")            : {"rows":798,  "target":TARGET,           "scaled":False},
    os.path.join(CLF_DIR, "train_clf.csv")                : {"rows":3724, "target":"quality_label_enc", "scaled":True},
    os.path.join(CLF_DIR, "val_clf.csv")                  : {"rows":796,  "target":"quality_label_enc", "scaled":True},
    os.path.join(CLF_DIR, "test_clf.csv")                 : {"rows":798,  "target":"quality_label_enc", "scaled":True},
    os.path.join(CLF_DIR, "train_clf_unscaled.csv")       : {"rows":3724, "target":"quality_label_enc", "scaled":False},
    os.path.join(CLF_DIR, "val_clf_unscaled.csv")         : {"rows":796,  "target":"quality_label_enc", "scaled":False},
    os.path.join(CLF_DIR, "test_clf_unscaled.csv")        : {"rows":798,  "target":"quality_label_enc", "scaled":False},
    os.path.join(CLU_DIR, "kmeans_full_scaled.csv")       : {"rows":5318, "target":TARGET,           "scaled":True},
}

all_ok = True
check_results = []
for fpath, spec in REQUIRED_FILES.items():
    fname = os.path.relpath(fpath, PROC_DIR).replace("\\", "/")
    ok = True
    issues = []
    if not os.path.exists(fpath):
        issues.append("FILE MISSING")
        ok = False
    else:
        d = pd.read_csv(fpath)
        if len(d) != spec["rows"]:
            issues.append(f"rows={len(d)} expected={spec['rows']}")
            ok = False
        if spec["target"] not in d.columns:
            issues.append(f"missing target col '{spec['target']}'")
            ok = False
        if d.isnull().sum().sum() > 0:
            issues.append(f"has NaN ({d.isnull().sum().sum()})")
            ok = False
        feat_cols = [c for c in d.columns if c != spec["target"]]
        if spec["scaled"]:
            mean_abs = d[feat_cols].mean().abs().max()
            if mean_abs > 0.5:
                issues.append(f"not scaled? mean_abs={mean_abs:.3f}")
        else:
            # Unscaled should have values in original range
            if d[feat_cols].abs().max().max() < 1:
                issues.append("possibly scaled (values < 1)?")

    status = "OK" if ok else "FAIL: " + "; ".join(issues)
    if not ok:
        all_ok = False
    check_results.append({"File": fname, "Status": status})
    icon = "OK" if ok else "!!"
    print(f"  [{icon}] {fname:<50} {status}")

# ══════════════════════════════════════════════════════════════════════
# 6. CLASS BALANCE REPORT
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  [6] CLASS BALANCE — Classification splits")
print("=" * 65)

for fname in ["train_clf.csv", "val_clf.csv", "test_clf.csv"]:
    d = pd.read_csv(os.path.join(PROC_DIR, fname))
    vc = d["quality_label_enc"].value_counts().sort_index()
    line = f"  {fname:<30}"
    for v, name in [(0,"low"),(1,"medium"),(2,"high")]:
        n   = vc.get(v, 0)
        pct = n / len(d) * 100
        line += f"  {name}={n}({pct:.1f}%)"
    print(line)

print("\n  NOTE: 'medium' class dominates (~76%) — imbalance handled by:")
print("    - LightGBM: SMOTE-Tomek applied in training script")
print("    - Other models: class_weight='balanced' or manual weighting")

# ══════════════════════════════════════════════════════════════════════
# 7. SAVE DATA SUMMARY
# ══════════════════════════════════════════════════════════════════════
summary_df = pd.DataFrame(check_results)
summary_path = os.path.join(PROC_DIR, "data_summary.csv")
summary_df.to_csv(summary_path, index=False)

print("\n" + "=" * 65)
print("  [7] FINAL FILE LIST")
print("=" * 65)

model_file_map = {
    "LINEAR REGRESSION"         : "train.csv, val.csv, test.csv",
    "POLYNOMIAL RIDGE"          : "train.csv, val.csv, test.csv",
    "KNN REGRESSION"            : "train.csv, val.csv, test.csv",
    "SVM REGRESSION (SVR)"      : "train.csv, val.csv, test.csv",
    "NEURAL NET REGRESSION"     : "train.csv, val.csv, test.csv",
    "DECISION TREE REGRESSION"  : "train_unscaled.csv, val_unscaled.csv, test_unscaled.csv",
    "RANDOM FOREST REGRESSION"  : "train_unscaled.csv, val_unscaled.csv, test_unscaled.csv",
    "LIGHTGBM REGRESSION"       : "train_unscaled.csv, val_unscaled.csv, test_unscaled.csv",
    "ENSEMBLE REGRESSION"       : "train.csv + train_unscaled.csv",
    "---"                       : "---",
    "LOGISTIC REGRESSION"       : "train_clf.csv, val_clf.csv, test_clf.csv",
    "KNN CLASSIFICATION"        : "train_clf.csv, val_clf.csv, test_clf.csv",
    "SVM CLASSIFICATION (SVC)"  : "train_clf.csv, val_clf.csv, test_clf.csv",
    "NEURAL NET CLASSIFICATION" : "train_clf.csv, val_clf.csv, test_clf.csv",
    "DECISION TREE CLF"         : "train_clf_unscaled.csv, val_clf_unscaled.csv, test_clf_unscaled.csv",
    "RANDOM FOREST CLF"         : "train_clf_unscaled.csv, val_clf_unscaled.csv, test_clf_unscaled.csv",
    "LIGHTGBM + SMOTE-TOMEK"    : "train_clf_unscaled.csv (SMOTE in-script), val_clf_unscaled.csv, test_clf_unscaled.csv",
    "ENSEMBLE CLASSIFICATION"   : "train_clf.csv + train_clf_unscaled.csv",
    "K-MEANS CLUSTERING"        : "kmeans_full_scaled.csv",
}

for model, files in model_file_map.items():
    if model == "---":
        print(f"  {'':->63}")
    else:
        print(f"  {model:<35} {files}")

print("\n" + "=" * 65)
if all_ok:
    print("  ALL FILES VERIFIED — DATA IS READY FOR ALL MODELS!")
else:
    print("  SOME FILES HAVE ISSUES — PLEASE CHECK ABOVE!")
print("=" * 65)
