# BÁO CÁO NGẮN – MLOPS WINE QUALITY

**Họ tên:** Trần Quang Trọng - 2A202601461  
**Cohort:** A20-K3

> Trạng thái: Bước 1 và Bước 2 đã hoàn thành. Bước 3 đã chạy xanh Unit Test/Train/Eval, nhưng Deploy còn lỗi khởi động service trên EC2.

## 1. Kết quả thực nghiệm Bước 1

Em huấn luyện trên `train_phase1.csv` (2.998 mẫu), đánh giá trên tập độc lập `eval.csv` (500 mẫu) và dùng MLflow ghi lại siêu tham số, accuracy, weighted F1-score cùng model artifact. Random Forest được giữ làm baseline; sau đó em mở rộng `train.py` bằng tham số `model_type` để thử nhiều thuật toán và ensemble (Bonus 2).

| Thực nghiệm | Runs thành công | Accuracy tốt nhất |
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

Em dùng DVC với Amazon S3 làm remote và EC2 Ubuntu 24.04 Free Tier làm máy phục vụ. Ở lần chạy thành công của Bước 2, GitHub Actions chạy đủ bốn job `Unit Test → Train → Eval → Deploy`; eval gate dùng ngưỡng mentor chấp nhận là 0.68. Model, metrics và dữ liệu DVC đã được upload lên S3.

Kết quả kiểm tra API sau deploy:

```text
GET  /health  -> {"status":"ok"}
POST /predict -> {"prediction":0,"label":"thap"}
```

Model đạt accuracy 0.6860, cao hơn ngưỡng 0.68 nên được phép triển khai. Security Group mở cổng 8000 giới hạn cho IP máy em để kiểm tra endpoint.

### 3.1. Kết quả Bước 3 – dữ liệu phase 2 và lỗi Deploy

Sau khi bổ sung dữ liệu phase 2, Unit Test, Train và Eval đều thành công; Deploy thất bại ở commit `1435c1f`. Log ghi nhận `systemctl restart --no-block mlops-serve`, sau đó `/health` không trả lời qua 15 lần kiểm tra và SSH action báo `remote command exited without exit status or exit signal`. Hai ảnh minh chứng là `screenshots/07_github_actions_deploy_failure_summary.png` và `screenshots/08_github_actions_deploy_health_timeout.png`.

Nguyên nhân đã được xác nhận bằng EC2 system console: kernel ghi `Out of memory: Killed process ... (python)` hai lần, với `anon-rss` khoảng 654--655 MiB. Model Extra Trees có kích thước khoảng 215 MB; khi đo trên máy local, riêng thao tác `joblib.load()` dùng khoảng 533 MiB RAM. Do EC2 là `t3.micro` với khoảng 1 GiB RAM, Linux OOM killer đã dừng tiến trình Python khi service nạp model. Vì vậy `/health` không lên và SSH action báo mất trạng thái remote command. Đây là lỗi giới hạn bộ nhớ của VM khi triển khai model lớn, không phải lỗi eval gate hay lỗi kết nối S3.

### 3.2. So sánh kết quả trước và sau khi bổ sung dữ liệu

Em chạy lại đúng cấu hình Extra Trees đã chốt trên tập đánh giá độc lập `eval.csv` (500 mẫu). Kết quả thực tế:

| Chỉ số | Bước 2: 2.998 mẫu | Bước 3: 5.996 mẫu | Thay đổi |
|---|---:|---:|---:|
| Accuracy | 0.6860 | **0.7640** | **+0.0780** |
| Weighted F1-score | 0.6841 | **0.7631** | **+0.0790** |

Việc tăng gấp đôi dữ liệu huấn luyện giúp cả accuracy và weighted F1-score tăng rõ rệt. Kết quả Bước 3 vượt ngưỡng eval gate `0.68`; Deploy vẫn chưa hoàn tất vì VM t3.micro thiếu RAM khi nạp model lớn, không phải vì chất lượng mô hình không đạt.
