"""
Anomaly Detector - phát hiện và phân loại các bất thường trong tín hiệu SCADA.

Phát hiện: noise spikes, missing samples, std deviation, IQR outliers,
signal drift, flatline, Isolation Forest, LOF.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Phát hiện và phân loại bất thường trong dữ liệu SCADA.

    Attributes:
        z_threshold (float): Ngưỡng Z-score cho spike detection.
        iqr_multiplier (float): Hệ số nhân IQR.
        missing_threshold (float): Tỉ lệ NaN/zero tối đa trước khi coi là lỗi.
    """

    # Nhãn loại bất thường
    ANOMALY_TYPES = {
        "spike": "Đột biến nhiễu",
        "missing": "Mất mẫu",
        "drift": "Drift dài hạn",
        "flatline": "Tín hiệu phẳng (sensor stuck)",
        "std_deviation": "Sai lệch chuẩn",
        "iqr_outlier": "IQR Outlier",
        "isolation_forest": "Isolation Forest",
        "lof": "Local Outlier Factor",
    }

    def __init__(
        self,
        z_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        missing_threshold: float = 0.05,
    ) -> None:
        """Khởi tạo AnomalyDetector.

        Args:
            z_threshold: Ngưỡng Z-score cho spike detection.
            iqr_multiplier: Hệ số nhân IQR cho outlier detection.
            missing_threshold: Tỉ lệ NaN/zero tối đa.
        """
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier
        self.missing_threshold = missing_threshold
        logger.debug(
            "AnomalyDetector khởi tạo: z=%.1f, iqr=%.1f, missing=%.2f",
            z_threshold, iqr_multiplier, missing_threshold,
        )

    # ------------------------------------------------------------------
    # Spike detection (Z-score)
    # ------------------------------------------------------------------

    def detect_noise_spikes(
        self,
        signal: Union[np.ndarray, pd.Series],
        z_threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Phát hiện đột biến nhiễu bằng Z-score.

        Args:
            signal: Tín hiệu đầu vào.
            z_threshold: Ngưỡng Z-score. Dùng self.z_threshold nếu None.

        Returns:
            Boolean array, True = bất thường.
        """
        z_threshold = z_threshold or self.z_threshold
        data = np.asarray(signal, dtype=float)
        mean = np.nanmean(data)
        std = np.nanstd(data)

        if std < 1e-12:
            return np.zeros(len(data), dtype=bool)

        z_scores = np.abs((data - mean) / std)
        anomalies = z_scores > z_threshold
        logger.debug(
            "Spike detection: %d/%d anomalies (z_threshold=%.1f)",
            np.sum(anomalies), len(data), z_threshold,
        )
        return anomalies

    # ------------------------------------------------------------------
    # Missing sample detection
    # ------------------------------------------------------------------

    def detect_missing_samples(
        self,
        signal: Union[np.ndarray, pd.Series],
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Phát hiện mất mẫu (NaN, zeros bất thường).

        Args:
            signal: Tín hiệu đầu vào.
            threshold: Tỉ lệ NaN tối đa. Dùng self.missing_threshold nếu None.

        Returns:
            Boolean array, True = mất mẫu (NaN hoặc zero bất thường).
        """
        threshold = threshold or self.missing_threshold
        data = np.asarray(signal, dtype=float)

        # NaN là mất mẫu rõ ràng
        nan_mask = np.isnan(data)

        # Zeros bất thường: zero khi giá trị xung quanh khác không
        zero_mask = data == 0.0
        non_nan_vals = data[~nan_mask]
        if len(non_nan_vals) > 0:
            non_zero_pct = np.mean(non_nan_vals != 0)
            if non_zero_pct > 0.95:
                # Hầu hết dữ liệu khác 0, nên zero là bất thường
                missing = nan_mask | zero_mask
            else:
                missing = nan_mask
        else:
            missing = nan_mask

        missing_rate = np.mean(missing)
        logger.debug(
            "Missing samples: %d (%.2f%%), threshold=%.2f",
            np.sum(missing), missing_rate * 100, threshold,
        )
        return missing

    # ------------------------------------------------------------------
    # Rolling std deviation
    # ------------------------------------------------------------------

    def detect_std_deviation(
        self,
        signal: Union[np.ndarray, pd.Series],
        window: int = 100,
        n_std: float = 3.0,
    ) -> np.ndarray:
        """Phát hiện sai lệch chuẩn rolling.

        Args:
            signal: Tín hiệu đầu vào.
            window: Kích thước cửa sổ rolling.
            n_std: Số lần std để xác định ngưỡng.

        Returns:
            Boolean array, True = bất thường.
        """
        series = pd.Series(np.asarray(signal, dtype=float))
        rolling_mean = series.rolling(window=window, center=True, min_periods=1).mean()
        rolling_std = series.rolling(window=window, center=True, min_periods=1).std()

        upper = rolling_mean + n_std * rolling_std
        lower = rolling_mean - n_std * rolling_std

        anomalies = (series > upper) | (series < lower)
        logger.debug(
            "Std deviation: %d anomalies, window=%d, n_std=%.1f",
            anomalies.sum(), window, n_std,
        )
        return anomalies.values

    # ------------------------------------------------------------------
    # IQR outliers
    # ------------------------------------------------------------------

    def detect_iqr_outliers(
        self,
        signal: Union[np.ndarray, pd.Series],
        multiplier: Optional[float] = None,
    ) -> np.ndarray:
        """Phát hiện outlier bằng IQR.

        Args:
            signal: Tín hiệu đầu vào.
            multiplier: Hệ số IQR. Dùng self.iqr_multiplier nếu None.

        Returns:
            Boolean array, True = outlier.
        """
        multiplier = multiplier or self.iqr_multiplier
        data = np.asarray(signal, dtype=float)

        q1 = np.nanpercentile(data, 25)
        q3 = np.nanpercentile(data, 75)
        iqr = q3 - q1

        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        outliers = (data < lower) | (data > upper)
        logger.debug(
            "IQR outliers: %d, bounds=[%.2f, %.2f]",
            np.sum(outliers), lower, upper,
        )
        return outliers

    # ------------------------------------------------------------------
    # Signal drift
    # ------------------------------------------------------------------

    def detect_signal_drift(
        self,
        signal: Union[np.ndarray, pd.Series],
        window: int = 500,
    ) -> np.ndarray:
        """Phát hiện drift dài hạn bằng so sánh moving average với global mean.

        Args:
            signal: Tín hiệu đầu vào.
            window: Kích thước cửa sổ rolling.

        Returns:
            Boolean array, True = drift.
        """
        data = np.asarray(signal, dtype=float)
        series = pd.Series(data)

        global_mean = np.nanmean(data)
        global_std = np.nanstd(data)

        if global_std < 1e-12:
            return np.zeros(len(data), dtype=bool)

        rolling_mean = series.rolling(window=window, min_periods=1).mean()
        drift_score = np.abs((rolling_mean - global_mean) / global_std)
        drift = (drift_score > self.z_threshold).values

        logger.debug("Signal drift: %d anomalies, window=%d", np.sum(drift), window)
        return drift

    # ------------------------------------------------------------------
    # Flatline detection
    # ------------------------------------------------------------------

    def detect_flatline(
        self,
        signal: Union[np.ndarray, pd.Series],
        window: int = 50,
        threshold: float = 0.001,
    ) -> np.ndarray:
        """Phát hiện tín hiệu phẳng bất thường (sensor stuck).

        Args:
            signal: Tín hiệu đầu vào.
            window: Kích thước cửa sổ kiểm tra.
            threshold: Ngưỡng std tối thiểu để không coi là flatline.

        Returns:
            Boolean array, True = flatline.
        """
        series = pd.Series(np.asarray(signal, dtype=float))
        rolling_std = series.rolling(window=window, min_periods=1).std()
        flatline = (rolling_std.fillna(0) < threshold).values

        logger.debug(
            "Flatline: %d anomalies, window=%d, threshold=%.4f",
            np.sum(flatline), window, threshold,
        )
        return flatline

    # ------------------------------------------------------------------
    # Isolation Forest
    # ------------------------------------------------------------------

    def detect_with_isolation_forest(
        self,
        df: pd.DataFrame,
        contamination: float = 0.05,
        random_state: int = 42,
    ) -> np.ndarray:
        """Phát hiện bất thường bằng Isolation Forest.

        Args:
            df: DataFrame với các features số.
            contamination: Tỉ lệ bất thường dự kiến.
            random_state: Seed ngẫu nhiên.

        Returns:
            Boolean array, True = bất thường.
        """
        # Chỉ lấy các cột số và điền NaN
        numeric_df = df.select_dtypes(include=[np.number]).fillna(df.select_dtypes(include=[np.number]).median())

        if numeric_df.empty:
            logger.warning("Không có cột số nào trong DataFrame")
            return np.zeros(len(df), dtype=bool)

        clf = IsolationForest(contamination=contamination, random_state=random_state)
        preds = clf.fit_predict(numeric_df.values)
        anomalies = preds == -1

        logger.info(
            "Isolation Forest: %d/%d anomalies (contamination=%.2f)",
            np.sum(anomalies), len(df), contamination,
        )
        return anomalies

    # ------------------------------------------------------------------
    # Local Outlier Factor
    # ------------------------------------------------------------------

    def detect_with_lof(
        self,
        df: pd.DataFrame,
        n_neighbors: int = 20,
    ) -> np.ndarray:
        """Phát hiện bất thường bằng Local Outlier Factor.

        Args:
            df: DataFrame với các features số.
            n_neighbors: Số láng giềng.

        Returns:
            Boolean array, True = bất thường.
        """
        numeric_df = df.select_dtypes(include=[np.number]).fillna(df.select_dtypes(include=[np.number]).median())

        if numeric_df.empty:
            logger.warning("Không có cột số nào trong DataFrame")
            return np.zeros(len(df), dtype=bool)

        n_neighbors = min(n_neighbors, len(numeric_df) - 1)
        clf = LocalOutlierFactor(n_neighbors=max(1, n_neighbors))
        preds = clf.fit_predict(numeric_df.values)
        anomalies = preds == -1

        logger.info(
            "LOF: %d/%d anomalies (n_neighbors=%d)",
            np.sum(anomalies), len(df), n_neighbors,
        )
        return anomalies

    # ------------------------------------------------------------------
    # Classify anomaly type
    # ------------------------------------------------------------------

    def classify_anomaly_type(
        self,
        signal: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """Phân loại loại bất thường cho từng điểm.

        Args:
            signal: Tín hiệu đầu vào.

        Returns:
            Array string với loại bất thường ('normal', 'spike', 'missing',
            'drift', 'flatline', 'std_deviation').
        """
        data = np.asarray(signal, dtype=float)
        n = len(data)
        labels = np.array(["normal"] * n, dtype=object)

        spikes = self.detect_noise_spikes(data)
        missing = self.detect_missing_samples(data)
        flatline = self.detect_flatline(data)
        drift = self.detect_signal_drift(data)
        std_dev = self.detect_std_deviation(data)

        # Ưu tiên nhãn (thứ tự từ thấp đến cao)
        labels[std_dev] = "std_deviation"
        labels[drift] = "drift"
        labels[flatline] = "flatline"
        labels[missing] = "missing"
        labels[spikes] = "spike"

        unique, counts = np.unique(labels, return_counts=True)
        logger.info("Anomaly types: %s", dict(zip(unique, counts)))
        return labels

    # ------------------------------------------------------------------
    # Anomaly report
    # ------------------------------------------------------------------

    def generate_anomaly_report(
        self,
        df: pd.DataFrame,
        signal_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Báo cáo tổng hợp tất cả anomalies.

        Args:
            df: DataFrame với DatetimeIndex hoặc integer index.
            signal_cols: Danh sách cột tín hiệu. None = tất cả cột số.

        Returns:
            DataFrame với các cột: timestamp, column, type, severity, value.
        """
        if signal_cols is None:
            signal_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        records = []
        for col in signal_cols:
            if col not in df.columns:
                continue
            signal = df[col].values
            labels = self.classify_anomaly_type(signal)

            # Xác định severity dựa trên loại bất thường
            severity_map = {
                "spike": "high",
                "missing": "high",
                "flatline": "medium",
                "drift": "medium",
                "std_deviation": "low",
                "iqr_outlier": "low",
            }

            anomaly_idx = np.where(labels != "normal")[0]
            for idx in anomaly_idx:
                ts = df.index[idx] if isinstance(df.index, pd.DatetimeIndex) else idx
                atype = labels[idx]
                records.append({
                    "timestamp": ts,
                    "column": col,
                    "type": atype,
                    "severity": severity_map.get(atype, "low"),
                    "value": float(signal[idx]) if not np.isnan(signal[idx]) else None,
                })

        report_df = pd.DataFrame(records)
        if not report_df.empty:
            report_df = report_df.sort_values("timestamp")

        logger.info(
            "Anomaly report: %d anomalies trên %d cột",
            len(report_df), len(signal_cols),
        )
        return report_df
