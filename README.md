# 💨 Wind Turbine SCADA Fault Detection System

End-to-end hệ thống phát hiện lỗi sớm trên tuabin gió sử dụng dữ liệu SCADA từ Kaggle.

## Kiến trúc Pipeline

```
Kaggle Dataset (Wind Turbine SCADA Data For Early Fault Detection)
     │
     ▼
[1] Data Ingestion Layer       ← Tải & lưu trữ dữ liệu
     │
     ▼
[2] Signal Processing Layer    ← Lọc nhiễu, FFT, kiểm tra liên tục
     │
     ▼
[3] Anomaly Detection Layer    ← Phát hiện nhiễu, mất mẫu, sai lệch, bất thường
     │
     ▼
[4] AI/ML Layer                ← Huấn luyện & phân loại dạng lỗi
     │
     ▼
[5] Dashboard (Streamlit)      ← Visualization & cảnh báo real-time
```

## Cấu trúc thư mục

```
Test_SCADA/
├── config/
│   └── config.yaml              ← Cấu hình toàn hệ thống
├── data/
│   ├── raw/                     ← Dataset gốc từ Kaggle
│   ├── processed/               ← Dữ liệu sau xử lý
│   └── models/                  ← Mô hình đã train
├── src/
│   ├── data_ingestion/
│   │   └── kaggle_downloader.py ← Tải & đọc dataset
│   ├── signal_processing/
│   │   ├── noise_filter.py      ← Butterworth, Wavelet, Kalman, MA, SG
│   │   ├── frequency_analysis.py← FFT, PSD, Spectrogram, STFT, THD
│   │   └── continuity_check.py  ← Kiểm tra tính liên tục dữ liệu
│   ├── anomaly_detection/
│   │   └── detector.py          ← Z-score, IQR, Isolation Forest, LOF
│   ├── ml_models/
│   │   ├── feature_engineering.py← Trích xuất đặc trưng
│   │   ├── fault_classifier.py  ← RF, XGB, SVM, GBM, Ensemble
│   │   └── lstm_autoencoder.py  ← LSTM Autoencoder (PyTorch/Keras)
│   └── dashboard/
│       └── app.py               ← Streamlit dashboard
├── notebooks/
│   └── exploratory_analysis.ipynb
├── tests/
│   ├── test_signal_processing.py
│   ├── test_anomaly_detection.py
│   └── test_ml_models.py
├── requirements.txt
├── setup.py
├── main.py                      ← CLI entry point
└── README.md
```

## Dataset

**Wind Turbine SCADA Data** từ Kaggle: `noobtube99/wind-turbine-scada-data`

| Cột | Mô tả | Đơn vị |
|-----|-------|--------|
| `Date/Time` | Timestamp | - |
| `LV ActivePower (kW)` | Công suất thực tế | kW |
| `Wind Speed (m/s)` | Tốc độ gió | m/s |
| `Theoretical_Power_Curve (KWh)` | Công suất lý thuyết | KWh |
| `Wind Direction (°)` | Hướng gió | ° |

- **Tần số lấy mẫu**: 10 phút
- **Thời gian**: ~1 năm

## Yêu cầu hệ thống

- Python >= 3.8
- RAM: >= 8 GB (cho LSTM training)
- GPU: tùy chọn (PyTorch/TensorFlow)

## Cài đặt

```bash
# Clone repository
git clone https://github.com/NgocAnizhere/Test_SCADA.git
cd Test_SCADA

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt

# Cài đặt package
pip install -e .
```

## Hướng dẫn sử dụng

### Bước 1: Tải dataset từ Kaggle

```bash
# Cấu hình Kaggle API credentials trước
# Xem: https://www.kaggle.com/docs/api
python main.py --mode download
```

### Bước 2: Xử lý tín hiệu

```bash
python main.py --mode process --input data/raw/ --output data/processed/
```

### Bước 3: Huấn luyện model

```bash
# Train Random Forest
python main.py --mode train --model rf

# Train XGBoost
python main.py --mode train --model xgb

# Train LSTM Autoencoder
python main.py --mode train --model lstm

# Train tất cả
python main.py --mode train --model all
```

### Bước 4: Phát hiện bất thường

```bash
python main.py --mode detect --input data/processed/
```

### Bước 5: Chạy Dashboard

```bash
python main.py --mode dashboard
# Hoặc trực tiếp:
streamlit run src/dashboard/app.py
```

Dashboard sẽ chạy tại: http://localhost:8501

## Mô tả module

### `src/data_ingestion/kaggle_downloader.py`
- `KaggleDownloader.download_dataset()`: Tải dataset từ Kaggle API
- `KaggleDownloader.load_data()`: Đọc CSV, parse datetime
- `KaggleDownloader.get_data_info()`: Thống kê dataset

### `src/signal_processing/noise_filter.py`
Bộ lọc nhiễu:
- **Butterworth lowpass/bandpass**: Lọc tần số
- **Wavelet denoising**: db4 với soft/hard thresholding
- **Kalman filter**: Lọc trạng thái tối ưu
- **Moving average**: Trung bình trượt
- **Savitzky-Golay**: Làm mịn giữ nguyên peak

### `src/signal_processing/frequency_analysis.py`
Phân tích tần số:
- FFT, PSD (Welch), Spectrogram, STFT
- Dominant frequencies, THD
- Phát hiện tần số bất thường

### `src/signal_processing/continuity_check.py`
Kiểm tra continuity:
- Missing timestamps, data gaps, duplicates
- Sampling rate consistency
- Interpolation (linear, cubic, ffill)
- Continuity score (0-100%)

### `src/anomaly_detection/detector.py`
Phát hiện bất thường:
- **Spike**: Z-score
- **Missing**: NaN/zero detection
- **Flatline**: Rolling std threshold
- **Drift**: Rolling mean vs global mean
- **IQR Outliers**: Interquartile range
- **Isolation Forest**: sklearn
- **LOF**: Local Outlier Factor

### `src/ml_models/fault_classifier.py`
Phân loại 6 dạng lỗi:
- `0`: Normal operation
- `1`: Sensor fault
- `2`: Electrical fault
- `3`: Mechanical fault
- `4`: Wind condition anomaly
- `5`: Communication fault

Algorithms: Random Forest, XGBoost, SVM, Gradient Boosting, Ensemble

### `src/ml_models/lstm_autoencoder.py`
LSTM Autoencoder (PyTorch hoặc Keras):
- Encoder: LSTM(128) → LSTM(64)
- Decoder: LSTM(64) → LSTM(128) → Dense
- Anomaly threshold: mean + 3×std reconstruction error

## Chạy Tests

```bash
# Chạy tất cả tests
pytest tests/ -v

# Chạy test riêng từng module
pytest tests/test_signal_processing.py -v
pytest tests/test_anomaly_detection.py -v
pytest tests/test_ml_models.py -v

# Chạy với coverage
pytest tests/ --cov=src --cov-report=html
```

## Kết quả kỳ vọng

| Metric | Kỳ vọng |
|--------|---------|
| Accuracy (RF) | > 85% |
| AUC-ROC | > 0.90 |
| LSTM Anomaly F1 | > 0.80 |
| Continuity Score | > 95% cho dữ liệu tốt |

## Dashboard Screenshots

> Chạy `streamlit run src/dashboard/app.py` để xem dashboard

Tab 1: **📊 Data Overview** - Thống kê, time series, missing data heatmap
Tab 2: **🔧 Signal Processing** - So sánh bộ lọc, FFT, Spectrogram
Tab 3: **🚨 Anomaly Detection** - Timeline, bảng anomalies, charts
Tab 4: **🤖 ML Classification** - Confusion matrix, ROC, feature importance
Tab 5: **📈 Reports** - Export Excel, trend analysis, alert history

## Troubleshooting

### Lỗi Kaggle authentication
```bash
# Tạo file ~/.kaggle/kaggle.json
mkdir ~/.kaggle
echo '{"username":"YOUR_USERNAME","key":"YOUR_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### Lỗi PyWavelets không có
```bash
pip install PyWavelets
```

### Lỗi CUDA/GPU
```bash
# Dùng CPU-only PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Memory error khi train LSTM
Giảm `sequence_length` và `batch_size` trong `config/config.yaml`.

## License

MIT License
