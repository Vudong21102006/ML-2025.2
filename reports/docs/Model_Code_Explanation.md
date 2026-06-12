# Giải Thích Chi Tiết Code — 5 Model Scripts

> **Dự án:** Wine Quality ML Pipeline  
> **Files được giải thích:**  
> `02_regression/SVR.py` | `03_classification/SVC.py` | `02_regression/RFR.py` | `03_classification/RFC.py` | `KMeans.py`

---

## Mục Lục

1. [Kiến Trúc Chung (Dùng Chung Cho Tất Cả Files)](#1-kiến-trúc-chung)
2. [SVR.py — Support Vector Regression](#2-svrpy--support-vector-regression)
3. [SVC.py — Support Vector Classification](#3-svcpy--support-vector-classification)
4. [RFR.py — Random Forest Regression](#4-rfrpy--random-forest-regression)
5. [RFC.py — Random Forest Classification](#5-rfcpy--random-forest-classification)
6. [KMeans.py — K-Means Clustering](#6-kmeanspy--k-means-clustering)
7. [Bảng Tổng Hợp — Kỹ Thuật Áp Dụng](#7-bảng-tổng-hợp)

---

## 1. Kiến Trúc Chung

### 1.1 Cấu trúc import path

```python
NOTEBOOKS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, NOTEBOOKS_DIR)
from utils import *
```

**Tại sao cần làm vậy?**  
Script `SVR.py` nằm tại `notebooks/02_regression/SVR.py`. Khi chạy lệnh `python notebooks/02_regression/SVR.py` từ thư mục gốc, Python không tự biết phải tìm `utils.py` ở đâu. Đoạn code này tính ngược 2 cấp lên để lấy thư mục `notebooks/`, rồi thêm vào `sys.path` (danh sách nơi Python tìm module).

- `__file__` = `d:/Project/ML-2025.2/notebooks/02_regression/SVR.py`
- `dirname(__file__)` = `.../02_regression/`
- `dirname(dirname(...))` = `.../notebooks/`

Sau đó `from utils import *` import tất cả hàm, hằng số và biến đường dẫn từ `utils.py`, bao gồm:

| Import từ utils | Loại | Dùng để |
|-----------------|------|---------|
| `load_regression_data()` | Hàm | Load data scaled cho SVR |
| `load_regression_data_unscaled()` | Hàm | Load data gốc cho RFR |
| `load_classification_data()` | Hàm | Load data scaled cho SVC |
| `load_classification_data_unscaled()` | Hàm | Load data gốc cho RFC |
| `load_kmeans_data()` | Hàm | Load toàn bộ dataset cho K-Means |
| `evaluate_regression()` | Hàm | Tính MAE, RMSE, R² |
| `evaluate_classification()` | Hàm | Tính Accuracy, F1-macro, F1-weighted |
| `save_model()` / `load_model()` | Hàm | Lưu/load model `.pkl` |
| `save_results()` | Hàm | Lưu metrics ra CSV |
| `plot_regression_scatter()` | Hàm | Vẽ Actual vs Predicted + Residual |
| `plot_confusion_matrix()` | Hàm | Vẽ Confusion Matrix |
| `plot_feature_importance()` | Hàm | Vẽ bar chart Feature Importance |
| `FIG_DIR`, `RESULTS_DIR`, `MODEL_DIR` | Hằng số | Đường dẫn lưu output |
| `LABEL_NAMES` | Hằng số | `["low", "medium", "high"]` |
| `r2_score`, `f1_score` | Hàm | Tính metric thủ công trong vòng lặp |

### 1.2 Cấu trúc bước chung (8 bước)

Mọi script đều tuân theo luồng **8 bước** nhất quán:

```
Bước 1: Load Data          → Kiểm tra kích thước, phân bố
Bước 2: Baseline           → Đánh giá với tham số mặc định (điểm tham chiếu)
Bước 3: Hyperparameter Tuning → GridSearch / RandomizedSearch
Bước 4: Evaluate Best Model → Train / Val / Test metrics
Bước 5: Cross-Validation   → Đánh giá ổn định trên Train+Val (5-fold)
Bước 6: Feature Analysis   → Importance hoặc Permutation Importance
Bước 7: Plots              → Trực quan hóa kết quả
Bước 8: Save               → Lưu model .pkl và results .csv
```

---

## 2. SVR.py — Support Vector Regression

**File:** `notebooks/02_regression/SVR.py`  
**Bài toán:** Dự đoán điểm chất lượng rượu (số nguyên 3–9)  
**Data:** `regression/train.csv` — StandardScaled

### 2.1 Tại sao SVR cần StandardScaled data?

SVM hoạt động bằng cách tìm **hyperplane** trong không gian đặc trưng, tối đa hóa margin. Khoảng cách trong không gian này được tính qua **Euclidean distance** giữa các điểm. Nếu feature `total sulfur dioxide` có giá trị 200 còn `pH` chỉ là 3.2, model sẽ bị thiên lệch về feature có scale lớn hơn.

StandardScaler chuyển mỗi feature về **mean=0, std=1** → mọi feature đóng góp công bằng.

### 2.2 Baseline SVR

```python
svr_base = SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale")
```

| Tham số | Giá trị mặc định | Ý nghĩa |
|---------|-----------------|---------|
| `kernel` | `"rbf"` | Radial Basis Function: chiếu dữ liệu lên không gian vô hạn chiều. Công thức: K(x,x') = exp(-γ‖x−x'‖²) |
| `C` | `1.0` | Hệ số phạt sai số. Lớn → fit sát training. Nhỏ → margin rộng hơn, regularization mạnh hơn |
| `epsilon` | `0.1` | Bán kính "ống epsilon-tube". Điểm nằm trong ống này **không bị phạt**. Model chỉ phạt các điểm vượt ra ngoài ống |
| `gamma` | `"scale"` | Tham số của kernel RBF. `"scale"` = 1/(n_features × Var(X)). Điều chỉnh "phạm vi ảnh hưởng" của mỗi điểm train |

**Mục tiêu của Baseline:** Có điểm tham chiếu ban đầu để so sánh sau khi tuning. Nếu baseline đã tốt rồi thì không cần tune nhiều.

### 2.3 GridSearchCV — Tìm siêu tham số tối ưu

```python
param_grid = {
    "C"      : [0.1, 1, 10, 100],      # 4 giá trị
    "epsilon": [0.05, 0.1, 0.2, 0.5],  # 4 giá trị
    "kernel" : ["rbf", "poly"],          # 2 giá trị
    "gamma"  : ["scale", "auto"],        # 2 giá trị
}
# → 4 × 4 × 2 × 2 = 64 tổ hợp × 5 folds = 320 lần fit
```

**GridSearchCV** thử **tất cả mọi tổ hợp** có thể. Với mỗi tổ hợp tham số, nó thực hiện **5-fold cross-validation** trên tập train:

```
Train set (3724 mẫu) chia thành 5 phần bằng nhau (~745 mẫu mỗi phần)
Fold 1: Train trên phần 2,3,4,5 → Test trên phần 1
Fold 2: Train trên phần 1,3,4,5 → Test trên phần 2
...
Fold 5: Train trên phần 1,2,3,4 → Test trên phần 5
→ Lấy trung bình 5 R² scores
```

Tham số `scoring="r2"` → tối ưu hóa R² (hệ số xác định).

**`refit=True`**: Sau khi tìm được best params, tự động fit lại model đó trên **toàn bộ** tập train (không phải chỉ 4/5).

**`n_jobs=-1`**: Chạy song song trên tất cả CPU cores.

### 2.4 Sensitivity Analysis — C vs R²

```python
C_values = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 200]
for c in C_values:
    m = SVR(kernel=best_kernel, C=c, epsilon=best_epsilon, gamma=best_gamma)
    m.fit(X_train, y_train)
    train_r2s.append(r2_score(y_train, m.predict(X_train)))
    val_r2s.append(r2_score(y_val, m.predict(X_val)))
```

Giữ **tất cả tham số khác cố định** ở giá trị best, chỉ thay đổi C. Điều này cho thấy:
- C nhỏ → cả Train và Val R² thấp → underfitting
- C vừa → Train và Val R² tốt, gap nhỏ → tốt
- C quá lớn → Train R² rất cao, Val R² giảm → overfitting

Đây là **bias-variance tradeoff** được trực quan hóa.

### 2.5 Kỹ thuật áp dụng trong SVR.py

| Kỹ thuật | Mô tả |
|----------|-------|
| **SVM với kernel RBF/Poly** | Chiếu dữ liệu lên không gian cao chiều để học phi tuyến |
| **Epsilon-tube (ε-SVR)** | Vùng không tính loss → robustness với noise nhỏ |
| **GridSearchCV** | Tìm siêu tham số tối ưu bằng brute-force |
| **5-fold Cross-Validation** | Đánh giá ổn định hơn single train-val split |
| **Sensitivity Analysis** | Trực quan hóa ảnh hưởng của C lên bias-variance |
| **StandardScaled input** | Bắt buộc cho SVM do tính chất distance-based |

---

## 3. SVC.py — Support Vector Classification

**File:** `notebooks/03_classification/SVC.py`  
**Bài toán:** Phân loại 3 lớp: 0=low / 1=medium / 2=high  
**Data:** `classification/train_clf.csv` — StandardScaled

### 3.1 Xử lý mất cân bằng lớp (Class Imbalance)

```python
svc_base = SVC(class_weight="balanced")
```

Tập train có phân bố:
```
low    (0):   166 mẫu  (4.5%)   ← lớp thiểu số
medium (1):  2852 mẫu  (76.6%)  ← lớp đa số
high   (2):   706 mẫu  (19.0%)
```

Nếu không xử lý, model sẽ luôn đoán "medium" để có accuracy cao (~77%) mà không học được gì về "low" và "high".

**`class_weight="balanced"`** tính trọng số tự động:

```
weight_i = n_samples / (n_classes × n_samples_i)

weight_low    = 3724 / (3 × 166)  = 7.48   → "low" được trọng số cao
weight_medium = 3724 / (3 × 2852) = 0.44   → "medium" bị giảm trọng số
weight_high   = 3724 / (3 × 706)  = 1.76   → "high" được tăng nhẹ
```

Trọng số này nhân vào hệ số phạt C cho từng mẫu khi tính loss.

### 3.2 Tại sao scoring="f1_macro" thay vì accuracy?

```python
grid = GridSearchCV(..., scoring="f1_macro")
```

**Accuracy** không phản ánh đúng khi imbalanced:
- Model đoán toàn bộ "medium" → Accuracy = 76.6% nhưng F1-macro = 0.25 (lớp low và high không được dự đoán đúng gì cả)

**F1-macro** tính F1 cho từng lớp rồi lấy trung bình không có trọng số:
```
F1_low    = 2 × (precision_low × recall_low) / (precision_low + recall_low)
F1_medium = ...
F1_high   = ...
F1_macro  = (F1_low + F1_medium + F1_high) / 3
```

→ Mỗi lớp được coi là quan trọng như nhau, kể cả lớp thiểu số.

### 3.3 Permutation Feature Importance

```python
from sklearn.inspection import permutation_importance
perm = permutation_importance(
    svc_best, X_val, y_val,
    n_repeats=15,
    scoring="f1_macro",
)
```

**Tại sao không dùng `feature_importances_`?**  
SVM không có thuộc tính `feature_importances_` như Random Forest. SVM học qua **support vectors** và **kernel trick**, không theo cơ chế cây quyết định.

**Permutation Importance** hoạt động như sau:
1. Tính F1-macro gốc trên Val set → `score_base`
2. Với mỗi feature i:
   - Xáo trộn (shuffle) ngẫu nhiên cột feature i → `score_shuffled`
   - `importance_i = score_base - score_shuffled`
3. Lặp lại `n_repeats=15` lần → lấy mean và std

**Ý nghĩa:** Feature quan trọng thì khi bị xáo trộn, performance giảm nhiều.

### 3.4 Kỹ thuật áp dụng trong SVC.py

| Kỹ thuật | Mô tả |
|----------|-------|
| **SVM Multi-class (One-vs-One)** | sklearn tự xử lý multi-class bằng OvO: tạo C(3,2)=3 binary classifiers |
| **class_weight='balanced'** | Xử lý imbalanced data bằng trọng số ngược tần suất |
| **probability=True** | Bật Platt scaling để có predict_proba() |
| **GridSearchCV + f1_macro** | Tối ưu cho imbalanced multi-class |
| **Permutation Feature Importance** | Đo importance cho model không có native feature_importances_ |
| **Sensitivity Analysis (C vs F1)** | Trực quan hóa bias-variance tradeoff |

---

## 4. RFR.py — Random Forest Regression

**File:** `notebooks/02_regression/RFR.py`  
**Bài toán:** Dự đoán điểm chất lượng rượu (3–9)  
**Data:** `regression/train_unscaled.csv` — **Không scale**

### 4.1 Tại sao Random Forest không cần StandardScaling?

Random Forest là ensemble của nhiều **Decision Tree**. Mỗi cây split node bằng cách chọn ngưỡng tốt nhất cho từng feature. Ngưỡng này là **relative** (so sánh tương đối) chứ không phải khoảng cách tuyệt đối.

```
Split node: alcohol <= 11.5 ?
  → YES: predict 5.2
  → NO:  predict 6.8
```

Dù `alcohol` là 11.5 hay 1.15 (sau MinMax scale), quyết định split vẫn như nhau về mặt thứ tự. Do đó **scaling không ảnh hưởng** đến tree-based models.

Dùng unscaled data còn có lợi: **interpretability** — feature importance và split thresholds có ý nghĩa vật lý thực tế.

### 4.2 OOB Score — Đánh giá "miễn phí"

```python
rfr_base = RandomForestRegressor(n_estimators=100, oob_score=True)
rfr_base.fit(X_train, y_train)
print(rfr_base.oob_score_)
```

Random Forest dùng **Bootstrap Aggregating (Bagging)**:
- Mỗi cây được train trên **mẫu bootstrap** = lấy ngẫu nhiên có hoàn lại từ train set
- Mỗi bootstrap trung bình dùng **~63.2%** samples (do có hoàn lại)
- **~36.8%** còn lại (Out-of-Bag) không được dùng để train cây đó

Với mỗi sample, lấy **trung bình dự đoán** từ các cây mà sample đó là OOB → so sánh với y thực → ra `oob_score_`.

**Lợi ích:** Ước lượng generalization error **không cần validation set riêng**, tương đương Leave-One-Out CV về mặt lý thuyết.

### 4.3 RandomizedSearchCV thay vì GridSearchCV

```python
rand_search = RandomizedSearchCV(
    estimator=RandomForestRegressor(...),
    param_distributions=param_dist,
    n_iter=40,       # Chỉ thử 40 tổ hợp ngẫu nhiên
    ...
)
```

**Tại sao không dùng GridSearch?**

```
GridSearch đầy đủ: 4 × 5 × 4 × 4 × 4 = 1,280 tổ hợp × 5 folds = 6,400 fits
RandomizedSearch:  40 tổ hợp × 5 folds = 200 fits → nhanh hơn ~32×
```

Nghiên cứu (Bergstra & Bengio, 2012) cho thấy RandomizedSearch **thường tìm được kết quả tương đương** GridSearch vì:
- Không gian tham số thực tế thường không đều (một số tham số quan trọng hơn)
- Sampling ngẫu nhiên explore nhiều vùng hơn GridSearch (không bị "cách đều")

### 4.4 Feature Importance — Gini / MDI

```python
rfr_best.feature_importances_
```

Random Forest tính **Mean Decrease in Impurity (MDI)**:
- Mỗi lần split node, impurity giảm đi một lượng
- Cộng dồn mức giảm đó cho từng feature trên toàn bộ cây và toàn bộ forest
- Chuẩn hóa để tổng = 1

**Với Regression:** Impurity = MSE (Mean Squared Error)  
→ Feature quan trọng = feature mà khi split theo nó, MSE giảm nhiều nhất

**Hạn chế của MDI:** Có thể thiên lệch về feature có nhiều unique values. Cần cẩn thận khi diễn giải.

### 4.5 Kỹ thuật áp dụng trong RFR.py

| Kỹ thuật | Mô tả |
|----------|-------|
| **Bootstrap Aggregating (Bagging)** | Mỗi cây train trên random subsample → giảm variance |
| **Feature Subsampling** | Mỗi split chỉ xem xét sqrt/log2 features ngẫu nhiên → decorrelate trees |
| **OOB Score** | Đánh giá generalization không cần val set |
| **RandomizedSearchCV** | Tìm hyperparameter hiệu quả hơn GridSearch cho không gian lớn |
| **MDI Feature Importance** | Đo tầm quan trọng feature qua mức giảm MSE |
| **Sensitivity Analysis (n_estimators)** | Kiểm tra convergence — khi nào thêm cây không cải thiện nữa |
| **Unscaled Input** | Đúng với đặc điểm tree-based model, giữ interpretability |

---

## 5. RFC.py — Random Forest Classification

**File:** `notebooks/03_classification/RFC.py`  
**Bài toán:** Phân loại 3 lớp: 0=low / 1=medium / 2=high  
**Data:** `classification/train_clf_unscaled.csv` — **Không scale**

### 5.1 class_weight='balanced' trong RFC

```python
rfc = RandomForestClassifier(class_weight="balanced")
```

Khác với RFR (regression, không cần trọng số), RFC (classification) cần xử lý imbalanced. **`class_weight='balanced'`** trong RFC hoạt động tại cấp độ từng cây:

Khi tính Gini Impurity để chọn split tốt nhất, **trọng số của từng mẫu** được nhân vào công thức:

```
Gini(node) = 1 - Σ (w_i × p_i)²
```

Nơi `w_i` là trọng số lớp của lớp i. → Lớp thiểu số "low" được ưu tiên hơn khi quyết định split.

### 5.2 predict_proba() — Xác suất dự đoán

```python
proba = rfc_best.predict_proba(X_test)  # shape: (798, 3)
```

Với Random Forest, xác suất của một mẫu thuộc lớp k được tính là:

```
P(class=k | x) = (1/n_trees) × Σ I(tree_i dự đoán class k cho x)
```

Đây là **majority vote** được làm mềm thành xác suất. Không như SVM (phải dùng Platt Scaling bổ sung), Random Forest có xác suất tự nhiên và thường **calibrated** tốt hơn.

Biểu đồ phân phối xác suất `rfc_proba_dist.png` cho thấy model có "tự tin" không — nếu histogram tập trung ở 0 và 1 là model rõ ràng, nếu tập trung ở 0.5 là model không chắc chắn.

### 5.3 So sánh RFC và SVC (cùng bài toán Classification)

| Tiêu chí | SVC | RFC |
|----------|-----|-----|
| **Input Data** | StandardScaled | Unscaled |
| **Cơ chế** | Hyperplane + kernel trick | Ensemble of Decision Trees |
| **Feature Importance** | Permutation Importance (gián tiếp) | MDI (trực tiếp từ model) |
| **predict_proba** | Platt Scaling (bổ sung) | Tự nhiên từ vote |
| **Hyperparameter tuning** | GridSearch (ít params) | RandomizedSearch (nhiều params) |
| **Thời gian train** | Nhanh hơn | Chậm hơn (nhiều cây) |
| **Interpretability** | Thấp (black box) | Cao hơn (feature importance) |

### 5.4 Kỹ thuật áp dụng trong RFC.py

| Kỹ thuật | Mô tả |
|----------|-------|
| **Random Forest Classifier** | Ensemble voting của nhiều Decision Trees |
| **class_weight='balanced'** | Xử lý imbalanced qua weighted Gini |
| **RandomizedSearchCV + f1_macro** | Tuning hiệu quả cho multi-class imbalanced |
| **OOB Score** | Tương tự RFR |
| **MDI Feature Importance** | Đo tầm quan trọng qua mức giảm Gini Impurity |
| **predict_proba histogram** | Phân tích mức độ "tự tin" của model |
| **Sensitivity: n_estimators vs F1** | Kiểm tra khi nào model bão hòa |

---

## 6. KMeans.py — K-Means Clustering

**File:** `notebooks/KMeans.py`  
**Bài toán:** Phân cụm (Unsupervised) — tìm cấu trúc tự nhiên trong dữ liệu  
**Data:** `clustering/kmeans_full_scaled.csv` — **5318 mẫu, StandardScaled**

### 6.1 Tại sao K-Means cần StandardScaled data?

Khác với Random Forest, K-Means tính **khoảng cách Euclidean** để gán mỗi điểm vào centroid gần nhất:

```
d(x, centroid) = √[Σ (x_i - centroid_i)²]
```

Nếu `total sulfur dioxide` có giá trị ~150 còn `pH` chỉ ~3.2, khoảng cách sẽ bị dominated hoàn toàn bởi SO₂. Tương tự như SVM, K-Means **bắt buộc** cần scaling.

### 6.2 init="k-means++"

```python
km = KMeans(n_clusters=k, init="k-means++", n_init=10)
```

**K-Means thông thường (random init):**
- Khởi tạo k centroids ngẫu nhiên từ dataset
- Có thể rơi vào **local minima** nếu centroids khởi đầu xấu

**K-Means++ (Arthur & Vassilvitskii, 2007):**
- Chọn centroid đầu tiên ngẫu nhiên
- Các centroid tiếp theo được chọn với **xác suất tỷ lệ với khoảng cách** đến centroid đã chọn
- Đảm bảo centroids ban đầu phân tán đều → **giảm đáng kể** nguy cơ local minima

`n_init=10`: Chạy toàn bộ K-Means 10 lần với seed khác nhau, lấy kết quả có Inertia thấp nhất.

### 6.3 Bốn metric đánh giá clustering

```python
inertia = km.inertia_                              # Within-cluster SS
sil     = silhouette_score(X, labels)              # [-1, 1]
db      = davies_bouldin_score(X, labels)          # ≥ 0
ch      = calinski_harabasz_score(X, labels)       # > 0
```

#### Inertia (Elbow Method)
```
Inertia = Σ_i Σ_{x ∈ C_i} ||x - μ_i||²
```
- Tổng bình phương khoảng cách từ mỗi điểm đến centroid của cụm nó
- **Nhỏ = cụm compact**, nhưng inertia luôn giảm khi tăng k → không đủ để chọn k tối ưu
- Dùng phương pháp **Elbow**: tìm điểm "gãy khuỷu" — nơi inertia giảm chậm lại đột ngột

#### Silhouette Score
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))

a(i) = khoảng cách TB từ điểm i đến các điểm khác trong cùng cụm (cohesion)
b(i) = khoảng cách TB từ điểm i đến cụm gần nhất khác (separation)
```
- Gần 1: điểm nằm đúng cụm, xa cụm khác
- Gần 0: điểm nằm sát ranh giới giữa 2 cụm
- Âm: điểm có thể đã bị gán nhầm cụm
- **Cao = tốt**

#### Davies-Bouldin Score
```
DB = (1/k) × Σ_i max_{j≠i} [(σ_i + σ_j) / d(μ_i, μ_j)]
```
- σ_i = độ phân tán trung bình trong cụm i
- d(μ_i, μ_j) = khoảng cách giữa 2 centroid
- **Nhỏ = tốt**: cụm compact và xa nhau

#### Calinski-Harabasz Score
```
CH = [SS_between / (k-1)] / [SS_within / (n-k)]
```
- SS_between: tổng phương sai giữa các cụm (between-cluster)
- SS_within: tổng phương sai trong các cụm (within-cluster)
- **Lớn = tốt**: cụm tách biệt rõ ràng và compact

### 6.4 Adjusted Rand Index — So sánh với nhãn thực tế

```python
ari = adjusted_rand_score(y_label, labels3)
nmi = normalized_mutual_info_score(y_label, labels3)
```

Đây là metric duy nhất **dùng nhãn ground truth** để đánh giá clustering.

**ARI (Adjusted Rand Index):**
- Đếm số cặp điểm được phân loại **nhất quán** giữa clustering và nhãn thực
- Điều chỉnh theo **expectation ngẫu nhiên** (random clustering)
- Giá trị: 0 = ngẫu nhiên, 1 = hoàn hảo, có thể âm

**NMI (Normalized Mutual Information):**
- Đo lượng thông tin chia sẻ giữa clustering và nhãn thực
- Chuẩn hóa về [0, 1]

### 6.5 Silhouette Plot

```python
sil_vals = silhouette_samples(X_arr, labels3)  # Silhouette của từng điểm
```

Biểu đồ này vẽ silhouette coefficient của **từng mẫu** (sắp xếp từ cao đến thấp trong mỗi cụm):
- Cụm tốt: hầu hết điểm có silhouette dương và tương đối cao
- Cụm xấu: nhiều điểm silhouette âm hoặc gần 0
- Các cụm nên có **độ rộng tương đương** nhau (không có cụm quá lớn/nhỏ so với k trung bình)

### 6.6 Radar Chart — Cluster Profile

```python
centers_norm = pd.DataFrame(km3.cluster_centers_, columns=feat_names)
```

`km3.cluster_centers_` là tọa độ (trong không gian StandardScaled) của k centroids. Vì data đã được chuẩn hóa về mean=0, std=1, centroid của cụm có alcohol cao sẽ có giá trị `alcohol` dương (trên trung bình).

Radar chart so sánh profile của 3 cụm trên 8 features, giúp nhận dạng cụm nào tương ứng với rượu "chất lượng cao" vs "chất lượng thấp".

### 6.7 Kỹ thuật áp dụng trong KMeans.py

| Kỹ thuật | Mô tả |
|----------|-------|
| **K-Means++ initialization** | Tránh local minima, ổn định hơn random init |
| **Multiple restarts (n_init=10)** | Chạy nhiều lần, lấy kết quả tốt nhất |
| **Elbow Method** | Trực quan hóa Inertia để chọn k |
| **Silhouette Score** | Metric đánh giá cluster quality tốt nhất trong 4 metric |
| **Davies-Bouldin Score** | Metric bổ sung (compact + separated) |
| **Calinski-Harabasz Score** | Metric bổ sung (between/within variance ratio) |
| **Adjusted Rand Index** | So sánh với nhãn ground truth |
| **Normalized Mutual Info** | Thêm góc nhìn thông tin lý thuyết |
| **Silhouette Plot** | Phân tích từng mẫu, không chỉ trung bình |
| **Radar Chart** | Trực quan hóa profile đặc trưng của từng cụm |
| **Contingency Table** | Xem clustering khớp với nhãn thực như thế nào |
| **Full dataset (5318 mẫu)** | Clustering không dùng train/val/test split |

---

## 7. Bảng Tổng Hợp — Kỹ Thuật Áp Dụng

| Kỹ thuật | SVR | SVC | RFR | RFC | KMeans |
|----------|:---:|:---:|:---:|:---:|:------:|
| StandardScaled input | ✅ | ✅ | ❌ | ❌ | ✅ |
| Unscaled input | ❌ | ❌ | ✅ | ✅ | ❌ |
| GridSearchCV | ✅ | ✅ | ❌ | ❌ | ❌ |
| RandomizedSearchCV | ❌ | ❌ | ✅ | ✅ | ❌ |
| 5-fold Cross-Validation | ✅ | ✅ | ✅ | ✅ | ❌ |
| OOB Score | ❌ | ❌ | ✅ | ✅ | ❌ |
| class_weight='balanced' | ❌ | ✅ | ❌ | ✅ | ❌ |
| MDI Feature Importance | ❌ | ❌ | ✅ | ✅ | ❌ |
| Permutation Importance | ❌ | ✅ | ❌ | ❌ | ❌ |
| Sensitivity Analysis | ✅ | ✅ | ✅ | ✅ | ❌ |
| Elbow Method | ❌ | ❌ | ❌ | ❌ | ✅ |
| Silhouette Score | ❌ | ❌ | ❌ | ❌ | ✅ |
| Davies-Bouldin | ❌ | ❌ | ❌ | ❌ | ✅ |
| Calinski-Harabasz | ❌ | ❌ | ❌ | ❌ | ✅ |
| Adjusted Rand Index | ❌ | ❌ | ❌ | ❌ | ✅ |
| predict_proba | ❌ | ✅ | ❌ | ✅ | ❌ |
| Actual vs Predicted plot | ✅ | ❌ | ✅ | ❌ | ❌ |
| Residual plot | ✅ | ❌ | ✅ | ❌ | ❌ |
| Confusion Matrix | ❌ | ✅ | ❌ | ✅ | ❌ |
| Contingency Table | ❌ | ❌ | ❌ | ❌ | ✅ |
| Radar Chart | ❌ | ❌ | ❌ | ❌ | ✅ |
| Silhouette Plot | ❌ | ❌ | ❌ | ❌ | ✅ |

### Metrics sử dụng

| Metric | Áp dụng | Công thức tóm tắt |
|--------|---------|-------------------|
| **MAE** | SVR, RFR | mean(\|y - ŷ\|) |
| **RMSE** | SVR, RFR | √mean((y-ŷ)²) |
| **R²** | SVR, RFR | 1 - SS_res/SS_tot |
| **Accuracy** | SVC, RFC | đúng/tổng |
| **F1-macro** | SVC, RFC | mean(F1 mỗi lớp) |
| **F1-weighted** | SVC, RFC | weighted mean(F1) theo n_samples |
| **Silhouette** | KMeans | (b-a)/max(a,b) |
| **Davies-Bouldin** | KMeans | mean(max separation ratio) |
| **Calinski-Harabasz** | KMeans | between/within variance ratio |
| **ARI** | KMeans | adjusted pair agreement vs ground truth |
