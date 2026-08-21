# BÁO CÁO NGẮN – MLOPS WINE QUALITY

**Họ tên:** Trần Quang Trọng - 2A202601461  
**Cohort:** A20-K3

> Trạng thái: Đã hoàn thành Bước 1 và Bước 2; sẽ bổ sung kết quả Bước 3 trước khi nộp.

## 1. Kết quả thực nghiệm Bước 1

Em huấn luyện trên `train_phase1.csv` (2.998 mẫu), đánh giá trên tập độc lập `eval.csv` (500 mẫu) và dùng MLflow ghi lại siêu tham số, accuracy, weighted F1-score cùng model artifact. Random Forest được giữ làm baseline; sau đó em mở rộng `train.py` bằng tham số `model_type` để thử nhiều thuật toán và ensemble (Bonus 2).

| Nhóm thực nghiệm | Runs thành công | Accuracy tốt nhất |
|---|---:|---:|
| Random Forest | 13 | 0.684 |
| Gradient Boosting | 3 | 0.632 |
| Extra Trees | 6 | **0.686** |
| Voting, stacking và calibration | 5 | **0.686** |
| HistGradientBoosting, XGBoost, CatBoost | 3 | 0.656 |
| Tổng | **30** | **0.686** |

Minh chứng gồm `screenshots/01_mlflow_30_runs.png` và `screenshots/02_mlflow_compare_models.png`. Hai ảnh cho thấy 30 runs thành công, các metrics, tham số và phần so sánh Extra Trees, Stacking, Random Forest. Có một run Voting thất bại do giới hạn multiprocessing của môi trường local; run này không được tính vào 30 runs thành công. Cấu hình cuối được em chọn là:

```yaml
model_type: extra_trees
n_estimators: 800
max_depth: null
min_samples_split: 2
min_samples_leaf: 1
max_features: sqrt
```

Extra Trees đạt accuracy **0.6860** và weighted F1-score **0.6841**. Stacking cũng đạt accuracy 0.686 với F1 cao hơn khoảng 0.0002, nhưng em chọn Extra Trees vì đơn giản hơn, train nhanh hơn và dễ triển khai trên VM. `params.yaml` đã được chốt theo cấu hình trên; `outputs/metrics.json` và `models/model.pkl` chứa kết quả/model cuối. Kết quả thấp hơn mốc tham chiếu 0.70 nhưng đã cải thiện baseline. Theo xác nhận của giảng viên, em cấu hình eval gate ở mức **0.68** để pipeline vẫn kiểm soát chất lượng và cho phép model 0.686 được triển khai.

## 2. Khó khăn và hướng giải quyết

Khó khăn chính là không mô hình nào vượt 0.70 dù đã thử nhiều cấu hình. Em so sánh công bằng trên cùng tập eval, giữ model tốt nhất và không đưa dữ liệu eval vào huấn luyện. Ngoài ra, MLflow 2.13 thiếu `pkg_resources` với setuptools mới nên em cố định `setuptools<82`; cổng 5000 bị macOS chiếm nên chuyển MLflow UI sang cổng 5050; Voting gặp lỗi multiprocessing nên em giới hạn `n_jobs=1` và chạy lại thành công.

## 3. Kết quả Bước 2 – CI/CD và triển khai AWS

Em dùng DVC với Amazon S3 làm remote và EC2 Ubuntu 24.04 Free Tier làm máy phục vụ. GitHub Actions đã chạy thành công đủ bốn job `Unit Test → Train → Eval → Deploy`; eval gate dùng ngưỡng mentor chấp nhận là 0.68. Lần chạy gần nhất có commit `6af3b30` và trạng thái Success. Model, metrics và report đã được upload lên S3; EC2 restart service tự động qua SSH.

Kết quả kiểm tra API sau deploy:

```text
GET  /health  -> {"status":"ok"}
POST /predict -> {"prediction":0,"label":"thap"}
```

Model đạt accuracy 0.6860, cao hơn ngưỡng 0.68 nên được triển khai. Security Group mở cổng 8000 giới hạn cho IP máy em để kiểm tra endpoint.
