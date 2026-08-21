# BÁO CÁO NGẮN – MLOPS WINE QUALITY

**Họ tên:** Trần Quang Trọng - 2A202601461  
**Cohort:** A20-K3

## 1. Kết quả thực nghiệm Bước 1

Em huấn luyện trên `train_phase1.csv` (2.998 mẫu), đánh giá trên `eval.csv` (500 mẫu) và dùng MLflow để ghi lại siêu tham số, metrics và model artifact. Em giữ Random Forest làm baseline, sau đó mở rộng `train.py` với `model_type` để thử nhiều thuật toán/ensemble (Bonus 2). MLflow ghi nhận 30 runs thành công; accuracy tốt nhất là 0.686.

| Nhóm model | Runs | Accuracy tốt nhất |
|---|---:|---:|
| Random Forest | 13 | 0.684 |
| Gradient Boosting | 3 | 0.632 |
| Extra Trees | 6 | **0.686** |
| Voting/stacking/calibration | 5 | 0.686 |
| HistGradientBoosting/XGBoost/CatBoost | 3 | 0.656 |

Cấu hình được chọn vì đạt kết quả cao, đơn giản và phù hợp triển khai:

```yaml
model_type: extra_trees
n_estimators: 800
max_depth: null
min_samples_split: 2
min_samples_leaf: 1
max_features: sqrt
```

Extra Trees đạt accuracy **0.6860**, weighted F1-score **0.6841**. Em chọn model này thay vì stacking vì dễ triển khai và train nhanh hơn. Minh chứng MLflow nằm tại `screenshots/01_mlflow_30_runs.png` và `screenshots/02_mlflow_compare_models.png`; kết quả/model cuối nằm trong `outputs/metrics.json` và `models/model.pkl`.

## 2. Khó khăn và cách giải quyết

Không model nào vượt mốc tham chiếu 0.70 dù đã thử nhiều cấu hình; em giữ cách đánh giá nhất quán trên cùng tập eval và không dùng eval để train. MLflow lỗi thiếu `pkg_resources` nên cố định `setuptools<82`; cổng 5000 bị chiếm nên dùng cổng 5050; Voting lỗi multiprocessing nên giới hạn `n_jobs=1`. Theo xác nhận của giảng viên, eval gate được đặt ở **0.68** để pipeline chấp nhận model 0.686; mốc README tham chiếu vẫn là 0.70.

## 3. Kết quả CI/CD và AWS

Em dùng DVC với Amazon S3 làm remote và EC2 Ubuntu 24.04 Free Tier để phục vụ API. Bước 2 đã chạy đủ `Unit Test → Train → Eval → Deploy`; model, metrics và dữ liệu DVC được đẩy lên S3. API kiểm tra thành công:

```text
GET  /health  -> {"status":"ok"}
POST /predict -> {"prediction":0,"label":"thap"}
```

Screenshot tương ứng: `03_github_actions_all_green.png`, `04_curl_api.png`, `05_s3_bucket_objects.png`, `06_s3_latest_model.png`. Bonus 3/5 có `outputs/report.txt` (confusion matrix, precision/recall) và phân phối nhãn trong `outputs/metrics.json`.

Sau khi bổ sung phase 2, Unit Test, Train và Eval thành công nhưng Deploy chưa hoàn tất. EC2 system log xác nhận kernel đã OOM-kill Python khi nạp model: model khoảng 215 MB, còn `t3.micro` chỉ có khoảng 1 GiB RAM. Vì vậy `/health` không khởi động và SSH action timeout; đây là giới hạn bộ nhớ VM, không phải lỗi eval gate hay S3. Minh chứng: `screenshots/07_github_actions_deploy_failure_summary.png` và `screenshots/08_github_actions_deploy_health_timeout.png`.

| Chỉ số | Bước 2 (2.998 mẫu) | Bước 3 (5.996 mẫu) | Thay đổi |
|---|---:|---:|---:|
| Accuracy | 0.6860 | **0.7640** | **+0.0780** |
| Weighted F1-score | 0.6841 | **0.7631** | **+0.0790** |

Việc tăng gấp đôi dữ liệu giúp cả accuracy và F1-score cải thiện rõ rệt; Bước 3 vượt gate 0.68. Tuy nhiên, Deploy Bước 3 còn bị giới hạn bởi RAM của EC2.
