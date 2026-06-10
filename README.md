# Wine Quality ML Pipeline — Hướng dẫn sử dụng

> **Dataset**: UCI Wine Quality (Red + White) — 5,318 mẫu sau tiền xử lý  
> **Tasks**: Regression (dự đoán điểm 3–9) + Classification (low / medium / high)

---

## Cấu trúc dự án

```
ML 2025.2/
├── data/
│   ├── raw/                              ← Dataset gốc (KHÔNG chỉnh sửa)
│   │   ├── winequality-red.csv
│   │   ├── winequality-white.csv
│   │   └── winequality.names
│   └── processed/
│       ├── regression/                   ← Dùng cho Regression models
│       │   ├── train.csv                 (3724 mẫu, StandardScaled)
│       │   ├── val.csv                   (796 mẫu,  StandardScaled)
│       │   ├── test.csv                  (798 mẫu,  StandardScaled)
│       │   ├── train_unscaled.csv        (3724 mẫu, Unscaled)
│       │   ├── val_unscaled.csv          (796 mẫu,  Unscaled)
│       │   └── test_unscaled.csv         (798 mẫu,  Unscaled)
│       ├── classification/               ← Dùng cho Classification models
│       │   ├── train_clf.csv             (3724 mẫu, StandardScaled)
│       │   ├── val_clf.csv               (796 mẫu,  StandardScaled)
│       │   ├── test_clf.csv              (798 mẫu,  StandardScaled)
│       │   ├── train_clf_unscaled.csv    (3724 mẫu, Unscaled)
│       │   ├── val_clf_unscaled.csv      (796 mẫu,  Unscaled)
│       │   └── test_clf_unscaled.csv     (798 mẫu,  Unscaled)
│       ├── clustering/                   ← Dùng cho K-means
│       │   └── kmeans_full_scaled.csv    (5318 mẫu, StandardScaled + cả 2 labels)
│       └── full/                         ← Dataset đầy đủ & các phiên bản scale khác
│           ├── wine_combined_processed.csv   (5318 mẫu, có FE, chưa scale)
│           ├── wine_minmax_scaled.csv        (5318 mẫu, MinMax [0,1])
│           ├── wine_robust_scaled.csv        (5318 mẫu, RobustScaler)
│           └── data_summary.csv
├── notebooks/
│   ├── utils.py                          ← ⭐ Module dùng chung — import vào mọi script
│   ├── 01_EDA_and_Preprocessing.py
│   ├── 02_prepare_additional_data.py     ← Chạy lại nếu cần tái tạo data
│   ├── 02_regression/                    ← Scripts cho Regression
│   └── 03_classification/                ← Scripts cho Classification
├── models/                               ← Model đã train (.pkl) được lưu ở đây
└── reports/
    ├── figures/                          ← Biểu đồ EDA + kết quả model
    └── results/                          ← Metrics (CSV) của từng model
```

---

## Thông tin về Data

### Features (15 cột — giống nhau cho mọi file)

| # | Feature | Mô tả | Đơn vị |
|---|---------|--------|--------|
| 1 | `fixed acidity` | Độ axit cố định | g/dm³ |
| 2 | `volatile acidity` | Độ axit bay hơi | g/dm³ |
| 3 | `citric acid` | Axit citric | g/dm³ |
| 4 | `residual sugar` | Đường còn lại | g/dm³ |
| 5 | `chlorides` | Chlorides | g/dm³ |
| 6 | `free sulfur dioxide` | SO₂ tự do | mg/dm³ |
| 7 | `total sulfur dioxide` | Tổng SO₂ | mg/dm³ |
| 8 | `density` | Mật độ | g/cm³ |
| 9 | `pH` | Độ pH | — |
| 10 | `sulphates` | Sulphates | g/dm³ |
| 11 | `alcohol` | Nồng độ cồn | % vol |
| 12 | `free_total_SO2_ratio` | **(FE)** Tỷ lệ SO₂ tự do / tổng | — |
| 13 | `acid_ratio` | **(FE)** Tỷ lệ fixed / volatile acidity | — |
| 14 | `sulphate_alcohol` | **(FE)** sulphates × alcohol | — |
| 15 | `is_red` | **(FE)** 1 = Red wine, 0 = White wine | binary |

> **FE** = Feature Engineering (tạo thêm từ features gốc)

### Target columns

| Cột | Giá trị | Dùng cho |
|-----|---------|---------|
| `quality` | Integer 3–9 | **Regression** |
| `quality_label_enc` | 0 = low (≤4), 1 = medium (5–6), 2 = high (≥7) | **Classification & Clustering** |

### Phân chia tập dữ liệu

| Tập | Số mẫu | Tỷ lệ | Ghi chú |
|-----|--------|-------|---------|
| Train | 3,724 | 70% | Dùng để huấn luyện |
| Validation | 796 | 15% | Dùng để tune hyperparameter |
| Test | 798 | 15% | Đánh giá cuối cùng — chỉ dùng 1 lần |

> **Stratified split** — tỷ lệ class giống nhau ở cả 3 tập

### Class balance (Classification)

```
low    (0):   166 /  3724 train  (4.5%)   → class thiểu số
medium (1):  2852 /  3724 train  (76.6%)  → class đa số
high   (2):   706 /  3724 train  (19.0%)
```

> ⚠️ Mất cân bằng lớp! Xem phần xử lý bên dưới.

---

## File nào dùng cho Model nào?

| Model | Task | File cần load |
|-------|------|--------------|
| **Linear Regression** | Regression | `regression/train.csv` (scaled) |
| **Polynomial Ridge Regression** | Regression | `regression/train.csv` (scaled) |
| **KNN Regression** | Regression | `regression/train.csv` (scaled) |
| **SVM / SVR** | Regression | `regression/train.csv` (scaled) |
| **Neural Network Regression** | Regression | `regression/train.csv` (scaled) |
| **Decision Tree Regression** | Regression | `regression/train_unscaled.csv` |
| **Random Forest Regression** | Regression | `regression/train_unscaled.csv` |
| **LightGBM Regression** | Regression | `regression/train_unscaled.csv` |
| **Ensemble Regression** | Regression | Tùy theo base models |
| **Logistic Regression** | Classification | `classification/train_clf.csv` (scaled) |
| **KNN Classification** | Classification | `classification/train_clf.csv` (scaled) |
| **SVM / SVC** | Classification | `classification/train_clf.csv` (scaled) |
| **Neural Network Classification** | Classification | `classification/train_clf.csv` (scaled) |
| **Decision Tree Classification** | Classification | `classification/train_clf_unscaled.csv` |
| **Random Forest Classification** | Classification | `classification/train_clf_unscaled.csv` |
| **LightGBM + SMOTE-Tomek** | Classification | `classification/train_clf_unscaled.csv` + SMOTE in-script |
| **Ensemble Classification** | Classification | Tùy theo base models |
| **K-means Clustering** | Clustering | `clustering/kmeans_full_scaled.csv` |

---

## Hướng dẫn Import — `utils.py`

> **Quan trọng**: Mọi script trong `02_regression/` và `03_classification/` đều phải thêm 2 dòng đầu này để import được `utils.py`.

### Bước 1 — Thêm đường dẫn vào sys.path

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import *
```

### Bước 2 — Load data theo đúng loại model

#### Regression — StandardScaled (Linear, Ridge, KNN, SVM, Neural Net)
```python
X_train, y_train, X_val, y_val, X_test, y_test = load_regression_data()
# X shape: (3724, 15), y range: 3-9
```

#### Regression — Unscaled (Decision Tree, Random Forest, LightGBM)
```python
X_train, y_train, X_val, y_val, X_test, y_test = load_regression_data_unscaled()
```

#### Classification — StandardScaled (Logistic, KNN, SVM, Neural Net)
```python
X_train, y_train, X_val, y_val, X_test, y_test = load_classification_data()
# y values: 0 (low), 1 (medium), 2 (high)
```

#### Classification — Unscaled (Decision Tree, Random Forest, LightGBM+SMOTE)
```python
X_train, y_train, X_val, y_val, X_test, y_test = load_classification_data_unscaled()
```

#### K-means Clustering
```python
X, y_quality, y_label = load_kmeans_data()
# X: 5318 mẫu, 15 features (StandardScaled)
# y_quality: điểm chất lượng gốc (3-9) — dùng để so sánh
# y_label: nhãn 3 lớp (0/1/2) — dùng để so sánh
```

---

## Các hàm hữu ích trong `utils.py`

### Evaluation

```python
# Regression
results = evaluate_regression(y_test, y_pred, split_name="Test")
# In ra: [Test] MAE=0.xxxx  RMSE=0.xxxx  R2=0.xxxx
# Trả về: {"MAE": ..., "RMSE": ..., "R2": ...}

# Classification
results = evaluate_classification(y_test, y_pred, split_name="Test")
# In ra: Accuracy, F1-macro, F1-weighted + Classification Report
# Trả về: {"Accuracy": ..., "F1_macro": ..., "F1_weighted": ...}
```

### Lưu model và kết quả

```python
save_model(model, "ten_model.pkl")     # Lưu vào models/
model = load_model("ten_model.pkl")    # Load lại từ models/

save_results({"Train": {...}, "Val": {...}, "Test": {...}}, "ten_model_results.csv")
# Lưu metrics ra reports/results/
```

### Vẽ biểu đồ

```python
# Regression: Scatter + Residual plot
plot_regression_scatter(y_test, y_pred,
    save_path=os.path.join(FIG_DIR, "ten_model_scatter.png"),
    title="Tên Model — Actual vs Predicted")

# Classification: Confusion Matrix
plot_confusion_matrix(y_test, y_pred,
    save_path=os.path.join(FIG_DIR, "ten_model_cm.png"),
    title="Tên Model — Confusion Matrix")

# Feature importance (cho tree-based models)
plot_feature_importance(model.feature_importances_,
    feature_names=list(X_train.columns),
    save_path=os.path.join(FIG_DIR, "ten_model_fi.png"))

# Hyperparameter tuning curve
plot_learning_curve(train_scores, val_scores,
    param_name="n_neighbors", param_values=[3,5,7,9,11],
    save_path=os.path.join(FIG_DIR, "ten_model_tuning.png"))
```

### Các hằng số tiện ích

```python
LABEL_NAMES   # ["low", "medium", "high"]
LABEL_COL     # "quality_label_enc"
TARGET        # "quality"
FIG_DIR       # Đường dẫn tới reports/figures/
RESULTS_DIR   # Đường dẫn tới reports/results/
MODEL_DIR     # Đường dẫn tới models/
```

---

## Template Script cơ bản

Copy và chỉnh sửa template này cho model của bạn:

```python
"""
Tên model — Regression / Classification
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import *

# Thay bằng sklearn model phù hợp
from sklearn.xxx import YourModel

SCRIPT = "0Xa_ten_model"
print_section(f"SCRIPT: {SCRIPT}")

# ── 1. Load data ───────────────────────────────────────────────────────
# Chọn 1 trong các hàm phù hợp:
X_train, y_train, X_val, y_val, X_test, y_test = load_regression_data()
# hoặc load_regression_data_unscaled()
# hoặc load_classification_data()
# hoặc load_classification_data_unscaled()

feat_names = list(X_train.columns)
print(f"  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

# ── 2. Train model ─────────────────────────────────────────────────────
model = YourModel(...)
model.fit(X_train, y_train)

# ── 3. Evaluate ────────────────────────────────────────────────────────
print_section("Evaluation")
results = {}
for split, X, y in [("Train", X_train, y_train),
                     ("Val",   X_val,   y_val),
                     ("Test",  X_test,  y_test)]:
    results[split] = evaluate_regression(y, model.predict(X), split_name=split)
    # hoặc evaluate_classification(...)

# ── 4. Plots ───────────────────────────────────────────────────────────
plot_regression_scatter(
    y_test, model.predict(X_test),
    save_path=os.path.join(FIG_DIR, f"{SCRIPT}_scatter.png"),
    title="Tên Model — Actual vs Predicted"
)

# ── 5. Save ────────────────────────────────────────────────────────────
save_model(model, f"{SCRIPT}.pkl")
save_results(results, f"{SCRIPT}_results.csv")

print_section("SUMMARY")
print(f"  Test: {results['Test']}")
print("  DONE!")
```

---

## Xử lý Class Imbalance

Với **Classification**, class `low` chỉ chiếm 4.5% → cần xử lý:

```python
# Cách 1: class_weight='balanced' (cho hầu hết sklearn models)
from sklearn.svm import SVC
model = SVC(class_weight='balanced')

# Cách 2: SMOTE-Tomek (chỉ dùng cho LightGBM theo quy ước của nhóm)
from imblearn.combine import SMOTETomek
X_train_res, y_train_res = SMOTETomek(random_state=42).fit_resample(X_train, y_train)
# Chỉ apply trên X_train — KHÔNG apply lên val/test!

# Cách 3: Dùng class_weight trong LightGBM
import lightgbm as lgb
model = lgb.LGBMClassifier(class_weight='balanced')
```

---

## Metrics đánh giá

### Regression
| Metric | Tốt khi | Ghi chú |
|--------|---------|---------|
| **MAE** | Thấp | Sai số trung bình tuyệt đối |
| **RMSE** | Thấp | Phạt nặng hơn với outlier |
| **R²** | Gần 1.0 | 1.0 = hoàn hảo, < 0 = tệ hơn đoán trung bình |

### Classification
| Metric | Tốt khi | Ghi chú |
|--------|---------|---------|
| **Accuracy** | Cao | Không đủ khi imbalanced |
| **F1-macro** | Cao | Quan trọng hơn khi mất cân bằng |
| **F1-weighted** | Cao | Tính trọng số theo số mẫu |

---

## Ghi chú quan trọng

> [!IMPORTANT]
> **Không chỉnh sửa** data ở `data/raw/`. Nếu cần tái tạo toàn bộ processed data, chạy theo thứ tự:
> ```
> python notebooks/01_EDA_and_Preprocessing.py
> python notebooks/02_prepare_additional_data.py
> ```

> [!WARNING]
> **Không apply SMOTE/oversampling lên val/test set** — chỉ apply trên train set trong script training.

> [!NOTE]
> **SMOTE-Tomek** chỉ dùng cho LightGBM Classification theo thỏa thuận của nhóm. Các model khác dùng `class_weight='balanced'` nếu cần.

> [!TIP]
> Khi chạy script từ thư mục con (`02_regression/`, `03_classification/`), luôn **chạy từ thư mục gốc** của project để tránh lỗi đường dẫn:
> ```
> # Đúng (từ thư mục gốc ML 2025.2/)
> python notebooks/02_regression/02b_knn_regression.py
>
> # Sai (từ bên trong thư mục con)
> cd notebooks/02_regression
> python 02b_knn_regression.py  ← có thể lỗi đường dẫn
> ```
