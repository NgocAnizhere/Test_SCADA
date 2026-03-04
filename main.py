"""
Main entry point - CLI để chạy các mode khác nhau.

Usage:
    python main.py --mode download
    python main.py --mode process --input data/raw/ --output data/processed/
    python main.py --mode train --model rf
    python main.py --mode detect --input data/processed/
    python main.py --mode dashboard
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

# Setup logging
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logging(log_level: str = "INFO", log_file: str = "logs/scada_system.log") -> None:
    """Cấu hình logging cho toàn hệ thống.

    Args:
        log_level: Mức log (DEBUG, INFO, WARNING, ERROR).
        log_file: Đường dẫn file log.
    """
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Tải cấu hình từ file YAML.

    Args:
        config_path: Đường dẫn file config.

    Returns:
        Dict cấu hình.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ==============================================================================
# Mode: download
# ==============================================================================

def run_download(config: dict) -> None:
    """Tải dataset từ Kaggle.

    Args:
        config: Dict cấu hình hệ thống.
    """
    logger = logging.getLogger("download")
    from src.data_ingestion.kaggle_downloader import KaggleDownloader

    raw_path = config.get("paths", {}).get("raw_data", "data/raw")
    downloader = KaggleDownloader(raw_data_path=raw_path)

    logger.info("Bắt đầu tải dataset...")
    success = downloader.download_dataset()

    if success:
        logger.info("Tải dataset thành công!")
        try:
            df = downloader.load_data()
            info = downloader.get_data_info(df)
            logger.info("Dataset info: shape=%s", info["shape"])
        except Exception as e:
            logger.warning("Không thể load dữ liệu sau khi tải: %s", e)
    else:
        logger.error("Tải dataset thất bại!")
        sys.exit(1)


# ==============================================================================
# Mode: process
# ==============================================================================

def run_process(config: dict, input_path: str, output_path: str) -> None:
    """Xử lý tín hiệu và kiểm tra continuity.

    Args:
        config: Dict cấu hình.
        input_path: Đường dẫn dữ liệu đầu vào.
        output_path: Đường dẫn lưu kết quả.
    """
    logger = logging.getLogger("process")
    import pandas as pd
    from src.data_ingestion.kaggle_downloader import KaggleDownloader
    from src.signal_processing.noise_filter import NoiseFilter
    from src.signal_processing.continuity_check import ContinuityChecker

    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Tải dữ liệu từ: %s", input_path)
    downloader = KaggleDownloader(raw_data_path=input_path)
    try:
        df = downloader.load_data()
    except FileNotFoundError:
        logger.error("Không tìm thấy file CSV trong: %s", input_path)
        sys.exit(1)

    sp_config = config.get("signal_processing", {}).get("noise_filter", {})
    fs = 1 / (config.get("data", {}).get("sampling_interval_minutes", 10) * 60)

    # Continuity check
    logger.info("Kiểm tra tính liên tục dữ liệu...")
    checker = ContinuityChecker(expected_freq="10min")
    report = checker.generate_continuity_report(df)
    logger.info("Continuity score: %.2f%%", report["continuity_score"])

    # Interpolate missing
    df_clean = checker.interpolate_missing(df, method="linear")

    # Apply filters
    noise_filter = NoiseFilter(fs=fs)
    numeric_cols = df_clean.select_dtypes(include=["number"]).columns.tolist()

    logger.info("Áp dụng bộ lọc Savitzky-Golay cho %d cột...", len(numeric_cols))
    df_filtered = df_clean.copy()

    for col in tqdm(numeric_cols, desc="Filtering"):
        try:
            filtered = noise_filter.savitzky_golay_filter(
                df_clean[col].values,
                window_length=sp_config.get("savitzky_golay_window", 11),
                polyorder=sp_config.get("savitzky_golay_polyorder", 3),
            )
            df_filtered[col] = filtered
        except Exception as e:
            logger.warning("Lỗi khi lọc cột %s: %s", col, e)

    # Lưu kết quả
    output_file = Path(output_path) / "processed_data.csv"
    df_filtered.to_csv(output_file)
    logger.info("Dữ liệu đã xử lý lưu tại: %s", output_file)


# ==============================================================================
# Mode: train
# ==============================================================================

def run_train(config: dict, model_type: str) -> None:
    """Huấn luyện model phân loại lỗi.

    Args:
        config: Dict cấu hình.
        model_type: Loại model ('rf', 'xgb', 'lstm', 'all').
    """
    logger = logging.getLogger("train")
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from src.ml_models.feature_engineering import FeatureEngineer
    from src.ml_models.fault_classifier import FaultClassifier
    from src.anomaly_detection.detector import AnomalyDetector

    processed_path = config.get("paths", {}).get("processed_data", "data/processed")
    models_path = config.get("paths", {}).get("models", "data/models")
    Path(models_path).mkdir(parents=True, exist_ok=True)

    # Load processed data
    processed_file = Path(processed_path) / "processed_data.csv"
    if not processed_file.exists():
        logger.error("Không tìm thấy dữ liệu đã xử lý. Chạy --mode process trước.")
        sys.exit(1)

    logger.info("Tải dữ liệu từ: %s", processed_file)
    df = pd.read_csv(processed_file, index_col=0, parse_dates=True)

    # Tạo nhãn tự động từ anomaly detection
    logger.info("Tạo nhãn lỗi tự động...")
    detector = AnomalyDetector()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    report = detector.generate_anomaly_report(df, signal_cols=numeric_cols[:4])

    labels = np.zeros(len(df), dtype=int)
    type_label_map = {
        "spike": 1, "flatline": 1,
        "drift": 2,
        "std_deviation": 3,
        "missing": 5,
    }

    if not report.empty:
        for _, row in tqdm(report.iterrows(), desc="Labeling", total=len(report)):
            try:
                idx = df.index.get_loc(row["timestamp"])
                atype = row.get("type", "")
                labels[idx] = type_label_map.get(atype, 0)
            except (KeyError, TypeError):
                pass

    logger.info(
        "Nhãn: %s",
        dict(zip(*np.unique(labels, return_counts=True)))
    )

    # Feature engineering
    logger.info("Trích xuất đặc trưng...")
    fe = FeatureEngineer(
        window_size=config.get("ml_models", {}).get("feature_engineering", {}).get("window_size", 100)
    )
    feature_df = fe.create_feature_matrix(df)

    # Align labels với feature matrix
    step = config.get("ml_models", {}).get("feature_engineering", {}).get("window_size", 100)
    y_labels = labels[step - 1::step][:len(feature_df)]

    # Drop non-numeric và timestamp columns
    feature_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    X = feature_df[feature_cols].fillna(0).values
    y = y_labels[:len(X)]

    ml_config = config.get("ml_models", {})
    test_size = ml_config.get("test_size", 0.2)
    random_state = ml_config.get("random_state", 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info("Train: %d, Test: %d samples", len(X_train), len(X_test))

    # Train models
    classifier = FaultClassifier(random_state=random_state)

    if model_type in ("rf", "all"):
        logger.info("Training Random Forest...")
        rf_model = classifier.train_random_forest(X_train, y_train)
        metrics = classifier.evaluate_model(rf_model, X_test, y_test)
        logger.info("RF accuracy: %.4f", metrics["accuracy"])
        classifier.save_model(rf_model, Path(models_path) / "random_forest.joblib")

    if model_type in ("xgb", "all"):
        logger.info("Training XGBoost...")
        try:
            xgb_model = classifier.train_xgboost(X_train, y_train)
            if xgb_model is not None:
                metrics = classifier.evaluate_model(xgb_model, X_test, y_test)
                logger.info("XGB accuracy: %.4f", metrics["accuracy"])
                classifier.save_model(xgb_model, Path(models_path) / "xgboost.joblib")
        except Exception as e:
            logger.warning("XGBoost training thất bại: %s", e)

    if model_type in ("lstm", "all"):
        logger.info("Training LSTM Autoencoder...")
        try:
            from src.ml_models.lstm_autoencoder import LSTMAutoencoder, BACKEND as LSTM_BACKEND
            if LSTM_BACKEND != "none":
                lstm = LSTMAutoencoder(
                    sequence_length=ml_config.get("lstm", {}).get("sequence_length", 48),
                    n_features=min(4, X.shape[1]),
                    encoding_dim=ml_config.get("lstm", {}).get("encoding_dim", 64),
                )
                lstm.build_model()
                # Tạo sequences từ dữ liệu bình thường
                seq_len = lstm.sequence_length
                X_normal = X_train[y_train == 0]
                if len(X_normal) > seq_len:
                    n_seq = (len(X_normal) - seq_len) // seq_len
                    X_seq = np.array([
                        X_normal[i * seq_len:(i + 1) * seq_len, :min(4, X.shape[1])]
                        for i in range(n_seq)
                        if (i + 1) * seq_len <= len(X_normal)
                    ])
                    if len(X_seq) > 0:
                        lstm.train(
                            X_seq,
                            epochs=ml_config.get("lstm", {}).get("epochs", 50),
                            batch_size=ml_config.get("lstm", {}).get("batch_size", 32),
                        )
                        lstm.set_threshold(X_seq)
                        lstm.save(Path(models_path) / "lstm_autoencoder.pt")
                        logger.info("LSTM Autoencoder đã train và lưu.")
        except Exception as e:
            logger.warning("LSTM training thất bại: %s", e)

    logger.info("Training hoàn tất! Models lưu tại: %s", models_path)


# ==============================================================================
# Mode: detect
# ==============================================================================

def run_detect(config: dict, input_path: str) -> None:
    """Phát hiện bất thường trên dữ liệu.

    Args:
        config: Dict cấu hình.
        input_path: Đường dẫn dữ liệu.
    """
    logger = logging.getLogger("detect")
    import pandas as pd
    from src.anomaly_detection.detector import AnomalyDetector

    # Tìm file CSV
    input_dir = Path(input_path)
    csv_files = list(input_dir.glob("*.csv"))
    if not csv_files:
        logger.error("Không tìm thấy file CSV trong: %s", input_path)
        sys.exit(1)

    df = pd.read_csv(csv_files[0], index_col=0, parse_dates=True)
    logger.info("Tải dữ liệu: %s, shape=%s", csv_files[0], df.shape)

    ad_config = config.get("anomaly_detection", {})
    detector = AnomalyDetector(
        z_threshold=ad_config.get("z_score_threshold", 3.0),
        iqr_multiplier=ad_config.get("iqr_multiplier", 1.5),
        missing_threshold=ad_config.get("missing_threshold", 0.05),
    )

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    logger.info("Chạy anomaly detection trên %d cột...", len(numeric_cols))

    report = detector.generate_anomaly_report(df, signal_cols=numeric_cols)
    logger.info("Phát hiện %d bất thường", len(report))

    if not report.empty:
        output_file = input_dir / "anomaly_report.csv"
        report.to_csv(output_file, index=False)
        logger.info("Báo cáo lưu tại: %s", output_file)

        # Summary
        by_type = report.groupby("type").size()
        by_severity = report.groupby("severity").size()
        logger.info("Theo loại:\n%s", by_type.to_string())
        logger.info("Theo mức độ:\n%s", by_severity.to_string())


# ==============================================================================
# Mode: dashboard
# ==============================================================================

def run_dashboard(config: dict) -> None:
    """Khởi chạy Streamlit dashboard.

    Args:
        config: Dict cấu hình.
    """
    logger = logging.getLogger("dashboard")
    import subprocess

    dashboard_config = config.get("dashboard", {})
    port = dashboard_config.get("port", 8501)
    host = dashboard_config.get("host", "localhost")

    app_path = Path(__file__).parent / "src" / "dashboard" / "app.py"
    logger.info("Khởi chạy dashboard tại http://%s:%d", host, port)

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(port),
        "--server.address", host,
    ]
    subprocess.run(cmd)


# ==============================================================================
# CLI argument parser
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Namespace với các arguments.
    """
    parser = argparse.ArgumentParser(
        description="Wind Turbine SCADA Fault Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python main.py --mode download
  python main.py --mode process --input data/raw/ --output data/processed/
  python main.py --mode train --model rf
  python main.py --mode train --model all
  python main.py --mode detect --input data/processed/
  python main.py --mode dashboard
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["download", "process", "train", "detect", "dashboard"],
        required=True,
        help="Chế độ chạy",
    )
    parser.add_argument(
        "--input",
        default="data/raw/",
        help="Đường dẫn dữ liệu đầu vào (cho process/detect)",
    )
    parser.add_argument(
        "--output",
        default="data/processed/",
        help="Đường dẫn lưu kết quả (cho process)",
    )
    parser.add_argument(
        "--model",
        choices=["rf", "xgb", "lstm", "all"],
        default="rf",
        help="Loại model để train",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Đường dẫn file config",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Mức độ log",
    )
    return parser.parse_args()


# ==============================================================================
# Main
# ==============================================================================

def main() -> None:
    """Entry point chính."""
    args = parse_args()
    setup_logging(log_level=args.log_level)
    logger = logging.getLogger("main")

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.warning("Không tìm thấy config file: %s. Dùng cấu hình mặc định.", args.config)
        config = {}

    logger.info("=== Wind Turbine SCADA Fault Detection System ===")
    logger.info("Mode: %s", args.mode)

    if args.mode == "download":
        run_download(config)
    elif args.mode == "process":
        run_process(config, args.input, args.output)
    elif args.mode == "train":
        run_train(config, args.model)
    elif args.mode == "detect":
        run_detect(config, args.input)
    elif args.mode == "dashboard":
        run_dashboard(config)

    logger.info("=== Hoàn tất ===")


if __name__ == "__main__":
    main()
