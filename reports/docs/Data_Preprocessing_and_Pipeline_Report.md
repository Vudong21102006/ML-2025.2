# Báo Cáo Chi Tiết: Xử Lý Dữ Liệu và Xây Dựng Data Pipeline (Wine Quality Dataset)

**Dự án:** ML 2025.2
**Tập dữ liệu:** UCI Wine Quality (Red & White Wine)
**Mục tiêu:** Chuẩn bị dữ liệu đầu vào chuẩn xác, toàn diện cho 2 bài toán chính: Regression (dự đoán điểm số chất lượng 3-9) và Classification (phân loại chất lượng: low/medium/high), cũng như bài toán Clustering.

---

## 1. Xử Lý Dữ Liệu (Data Preprocessing)

Toàn bộ quá trình tiền xử lý được thực hiện tự động thông qua script `notebooks/01_EDA_and_Preprocessing.py`. Các vấn đề về dữ liệu đã được phát hiện và giải quyết triệt để:

### 1.1 Khắc phục lỗi định dạng CSV gốc
Tập dữ liệu gốc có một đặc thù định dạng: header chứa các ký tự bọc ngoặc kép `""` làm ký tự thoát (escape). Nếu dùng `pd.read_csv()` mặc định, Pandas sẽ parse sai cấu trúc cột.
**Cách xử lý:** Đã viết một hàm helper `read_wine_csv()` mở file, đọc thủ công dòng đầu tiên, loại bỏ các ký tự escape dư thừa, sau đó dùng `io.StringIO` để parse phần dữ liệu còn lại một cách chính xác.

### 1.2 Loại bỏ dữ liệu trùng lặp (Deduplication)
Kiểm tra cho thấy tập dữ liệu có tỷ lệ trùng lặp khá cao.
*   **Kích thước ban đầu:** 6,497 mẫu.
*   **Thực hiện:** Đã loại bỏ 1,179 dòng bị trùng lặp chính xác trên toàn bộ các cột (bao gồm cả label).
*   **Kích thước sau xử lý:** 5,318 mẫu dữ liệu sạch.

### 1.3 Xử lý ngoại lệ (Outlier Handling)
Phân tích bằng phương pháp IQR và boxplot cho thấy một số biến (như `residual sugar`, `chlorides`, `free sulfur dioxide`) có các giá trị cực đoan, đuôi phân phối dài (skewed).
*   **Cách xử lý:** Sử dụng kỹ thuật **Winsorization** (clip giá trị tại phân vị 1% và 99%).
*   **Lý do:** Cách này giúp giới hạn ảnh hưởng của các ngoại lệ cực đoan mà không làm mất đi các mẫu dữ liệu quý giá (điều rất quan trọng vì tập dữ liệu vốn đã bị mất cân bằng lớp ở các nhóm quality cực tiểu/cực đại).

### 1.4 Trích xuất đặc trưng mới (Feature Engineering - FE)
Dựa trên kiến thức về thành phần hóa học của rượu, 4 đặc trưng mới (features) đã được tạo ra để giúp các mô hình ML học được các mối quan hệ phi tuyến tính tốt hơn:
1.  `free_total_SO2_ratio`: Tỷ lệ giữa lượng SO₂ tự do và tổng lượng SO₂.
2.  `acid_ratio`: Tỷ lệ giữa `fixed acidity` và `volatile acidity`.
3.  `sulphate_alcohol`: Tương tác nhân (interaction term) giữa `sulphates` và `alcohol` (đây là 2 biến có tương quan khá tốt với chất lượng).
4.  `is_red`: Biến cờ (binary) đánh dấu loại rượu (1 = Đỏ, 0 = Trắng).

**Tổng số features:** 15 features (11 gốc + 4 mới).

### 1.5 Thiết lập Target Labels
*   **Regression Target:** Giữ nguyên cột `quality` (kiểu số nguyên từ 3 đến 9).
*   **Classification Target:** Tạo thêm cột `quality_label_enc` chia quality thành 3 nhóm để giảm thiểu độ khó do mất cân bằng lớp cực đoan:
    *   `0 (low)`: quality ≤ 4
    *   `1 (medium)`: quality = 5, 6
    *   `2 (high)`: quality ≥ 7

---

## 2. Phân Chia Tập Dữ Liệu (Data Splitting)

Dữ liệu được chia thành 3 tập để phục vụ huấn luyện, tinh chỉnh (tuning) và đánh giá khách quan. Quá trình chia được thực hiện bằng `train_test_split` của thư viện scikit-learn với tham số `stratify` dựa trên cột `quality`.

*   **Quy mô:**
    *   **Train Set (70%):** 3,724 mẫu
    *   **Validation Set (15%):** 796 mẫu
    *   **Test Set (15%):** 798 mẫu
*   **Tính phân tầng (Stratified):** Đảm bảo tỷ lệ các lớp quality (ví dụ Q3, Q9) được giữ đồng đều trong cả 3 tập, ngăn chặn tình trạng một tập dữ liệu hoàn toàn vắng bóng một lớp thiểu số.

---

## 3. Chuẩn Hóa Dữ Liệu (Data Scaling) và Tổ Chức File

Tùy vào thuật toán ML mà dữ liệu cần hoặc không cần chuẩn hóa. Việc chuẩn hóa (`StandardScaler`) **chỉ được fit trên tập Train**, sau đó transform cho Val và Test để tránh hiện tượng rò rỉ dữ liệu (Data Leakage).

Script `notebooks/02_prepare_additional_data.py` đã tạo ra các bộ dữ liệu chuyên biệt và tự động sắp xếp vào thư mục `data/processed/` thành 4 nhóm rõ ràng:

1.  **`regression/` (Target: `quality`)**
    *   `train.csv`, `val.csv`, `test.csv`: Đã chuẩn hóa bằng StandardScaler (Z-score). Dùng cho: Linear, Ridge, KNN, SVM (SVR), Neural Net.
    *   `train_unscaled.csv`, `val_unscaled.csv`, `test_unscaled.csv`: Giữ nguyên giá trị gốc. Dùng cho: Decision Tree, Random Forest, LightGBM.
2.  **`classification/` (Target: `quality_label_enc` - 0, 1, 2)**
    *   `train_clf.csv`, `val_clf.csv`, `test_clf.csv`: Đã chuẩn hóa StandardScaler. Dùng cho: Logistic Regression, KNN, SVM (SVC), Neural Net.
    *   `train_clf_unscaled.csv`, `val_clf_unscaled.csv`, `test_clf_unscaled.csv`: Unscaled. Dùng cho Tree-based models. Riêng mô hình LightGBM sẽ sử dụng tập này kết hợp với thuật toán SMOTE-Tomek ngay trong script huấn luyện để xử lý vấn đề mất cân bằng lớp.
3.  **`clustering/`**
    *   `kmeans_full_scaled.csv`: Chứa toàn bộ 5,318 mẫu, đã được scale bằng StandardScaler, tích hợp sẵn cả target regression và classification để làm ground truth khi đánh giá phân cụm.
4.  **`full/`**
    *   Chứa tập dữ liệu gộp chưa scale (`wine_combined_processed.csv`), các phiên bản scale dự phòng (`MinMax`, `RobustScaler`) và `data_summary.csv` báo cáo chi tiết về từng file được tạo.

---

## 4. Xây Dựng Shared Utilities (`notebooks/utils.py`)

Để đảm bảo tính nhất quán, giảm thiểu việc lặp code (DRY - Don't Repeat Yourself) khi làm việc nhóm trên nhiều mô hình khác nhau, toàn bộ các hàm dùng chung đã được gom vào `utils.py`. Các thành viên nhóm chỉ cần import module này để thực hiện hầu hết các tác vụ phụ trợ.

Các nhóm chức năng chính trong `utils.py`:

### 4.1. Data Loaders (Nạp dữ liệu)
Định tuyến sẵn file dữ liệu từ các thư mục con ở phần 3 để lấy đúng features (X) và target (y).
*   `load_regression_data()` / `load_regression_data_unscaled()`
*   `load_classification_data()` / `load_classification_data_unscaled()`
*   `load_kmeans_data()`

### 4.2. Evaluation Functions (Đánh giá)
*   **`evaluate_regression(y_true, y_pred)`:** Trả về dict chứa các chỉ số **MAE**, **RMSE**, và **R²**. In kết quả trực quan ra console.
*   **`evaluate_classification(y_true, y_pred)`:** Trả về dict chứa **Accuracy**, **F1-macro**, **F1-weighted**. Tự động in ra Classification Report chi tiết của scikit-learn.

### 4.3. Visualization Helpers (Trực quan hóa Model)
Được cấu hình sẵn dùng `seaborn` và xuất ra dưới định dạng file ảnh chất lượng cao vào `reports/figures/`.
*   **`plot_regression_scatter()`:** Vẽ hai biểu đồ cạnh nhau: Actual vs Predicted (so sánh giá trị thực và dự đoán) và Residual plot (phân tích sai số).
*   **`plot_confusion_matrix()`:** Vẽ ma trận nhầm lẫn dạng Heatmap cho bài toán Classification.
*   **`plot_feature_importance()`:** Trực quan hóa Bar chart mức độ quan trọng của các đặc trưng (dùng cho các mô hình dạng Tree).
*   **`plot_learning_curve()`:** Trực quan hóa Line chart so sánh Train vs Val score khi thực hiện tinh chỉnh siêu tham số (hyperparameter tuning).

### 4.4. Persistence (Lưu trữ Model & Results)
*   **`save_model()` / `load_model()`:** Dump và load model objects bằng `pickle` vào/từ thư mục `models/`.
*   **`save_results()`:** Lưu lịch sử đánh giá metric của các mô hình thành file CSV vào `reports/results/` để dễ dàng làm thống kê so sánh tổng hợp sau này.

---

## 5. Kết Luận

Hệ thống Data Pipeline đã hoàn thiện quy trình End-to-End từ Raw CSV sang các Features sạch, sẵn sàng (Ready-to-Train). Việc tách bạch các dạng file Scale/Unscaled, Regression/Classification và tích hợp module Utilities mạnh mẽ giúp cả nhóm có thể tiến hành xây dựng và tinh chỉnh model ngay lập tức mà không phải bận tâm về việc chuẩn bị hay xử lý dữ liệu chồng chéo. Mọi rủi ro về Data Leakage hay sai lệch đường dẫn đã được loại trừ.
