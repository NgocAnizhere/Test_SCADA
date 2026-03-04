"""
Feature Engineer - trích xuất đặc trưng từ dữ liệu SCADA.

Bao gồm time-domain, frequency-domain, time-frequency và statistical features.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Trích xuất và lựa chọn đặc trưng từ tín hiệu SCADA.

    Attributes:
        window_size (int): Kích thước cửa sổ sliding window.
        rolling_windows (List[int]): Danh sách kích thước rolling windows.
        fs (float): Tần số lấy mẫu (Hz).
    """

    def __init__(
        self,
        window_size: int = 100,
        rolling_windows: Optional[List[int]] = None,
        fs: float = 1 / 600.0,
    ) -> None:
        """Khởi tạo FeatureEngineer.

        Args:
            window_size: Kích thước cửa sổ cho sliding window features.
            rolling_windows: Danh sách window sizes cho rolling features.
            fs: Tần số lấy mẫu.
        """
        self.window_size = window_size
        self.rolling_windows = rolling_windows or [10, 50, 100]
        self.fs = fs
        self._scaler: Optional[object] = None
        logger.debug(
            "FeatureEngineer khởi tạo: window=%d, rolling=%s",
            window_size, self.rolling_windows,
        )

    # ------------------------------------------------------------------
    # Time-domain features
    # ------------------------------------------------------------------

    def extract_time_domain_features(
        self,
        data: Union[np.ndarray, pd.Series],
    ) -> Dict[str, float]:
        """Trích xuất đặc trưng miền thời gian.

        Args:
            data: Tín hiệu đầu vào.

        Returns:
            Dict các đặc trưng: mean, std, min, max, rms, kurtosis, skewness,
            peak_to_peak, crest_factor, shape_factor.
        """
        arr = np.asarray(data, dtype=float)
        arr_clean = arr[~np.isnan(arr)]
        if len(arr_clean) == 0:
            return {k: 0.0 for k in [
                "mean", "std", "min", "max", "rms", "kurtosis",
                "skewness", "peak_to_peak", "crest_factor", "shape_factor"
            ]}

        rms = np.sqrt(np.mean(arr_clean ** 2))
        peak = np.max(np.abs(arr_clean))
        mean_abs = np.mean(np.abs(arr_clean))

        return {
            "mean": float(np.mean(arr_clean)),
            "std": float(np.std(arr_clean)),
            "min": float(np.min(arr_clean)),
            "max": float(np.max(arr_clean)),
            "rms": float(rms),
            "kurtosis": float(stats.kurtosis(arr_clean)),
            "skewness": float(stats.skew(arr_clean)),
            "peak_to_peak": float(np.max(arr_clean) - np.min(arr_clean)),
            "crest_factor": float(peak / rms) if rms > 1e-12 else 0.0,
            "shape_factor": float(rms / mean_abs) if mean_abs > 1e-12 else 0.0,
        }

    # ------------------------------------------------------------------
    # Frequency-domain features
    # ------------------------------------------------------------------

    def extract_frequency_domain_features(
        self,
        data: Union[np.ndarray, pd.Series],
        fs: Optional[float] = None,
    ) -> Dict[str, float]:
        """Trích xuất đặc trưng miền tần số từ FFT.

        Args:
            data: Tín hiệu đầu vào.
            fs: Tần số lấy mẫu. Dùng self.fs nếu None.

        Returns:
            Dict các đặc trưng: spectral_centroid, spectral_bandwidth,
            spectral_rolloff, dominant_freq, spectral_entropy.
        """
        fs = fs or self.fs
        arr = np.asarray(data, dtype=float)
        arr_clean = pd.Series(arr).ffill().fillna(0).values

        if len(arr_clean) < 4:
            return {k: 0.0 for k in [
                "spectral_centroid", "spectral_bandwidth",
                "spectral_rolloff", "dominant_freq", "spectral_entropy"
            ]}

        # FFT
        fft_vals = np.abs(np.fft.rfft(arr_clean))
        freqs = np.fft.rfftfreq(len(arr_clean), d=1.0 / fs)

        # Tránh chia cho 0
        total_power = np.sum(fft_vals ** 2)
        if total_power < 1e-12:
            return {k: 0.0 for k in [
                "spectral_centroid", "spectral_bandwidth",
                "spectral_rolloff", "dominant_freq", "spectral_entropy"
            ]}

        # Normalized power spectrum
        psd_norm = fft_vals ** 2 / total_power

        centroid = np.sum(freqs * psd_norm)
        bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * psd_norm))

        # Spectral rolloff (85% năng lượng)
        cumsum = np.cumsum(psd_norm)
        rolloff_idx = np.searchsorted(cumsum, 0.85)
        rolloff = freqs[rolloff_idx] if rolloff_idx < len(freqs) else freqs[-1]

        dominant_freq = freqs[np.argmax(fft_vals[1:])] if len(fft_vals) > 1 else 0.0

        # Spectral entropy
        psd_norm_safe = psd_norm + 1e-12
        entropy = -np.sum(psd_norm_safe * np.log2(psd_norm_safe))

        return {
            "spectral_centroid": float(centroid),
            "spectral_bandwidth": float(bandwidth),
            "spectral_rolloff": float(rolloff),
            "dominant_freq": float(dominant_freq),
            "spectral_entropy": float(entropy),
        }

    # ------------------------------------------------------------------
    # Time-frequency features (Wavelet energy)
    # ------------------------------------------------------------------

    def extract_wavelet_features(
        self,
        data: Union[np.ndarray, pd.Series],
        wavelet: str = "db4",
        levels: int = 5,
    ) -> Dict[str, float]:
        """Trích xuất đặc trưng wavelet energy theo mức.

        Args:
            data: Tín hiệu đầu vào.
            wavelet: Loại wavelet.
            levels: Số mức phân tích (1-5).

        Returns:
            Dict với wavelet_energy_level1 đến wavelet_energy_level{levels}.
        """
        arr = np.asarray(data, dtype=float)
        arr_clean = np.array(pd.Series(arr).fillna(0).values, dtype=float)  # writable copy

        features: Dict[str, float] = {}

        try:
            import pywt
            actual_level = min(levels, pywt.dwt_max_level(len(arr_clean), wavelet))
            coeffs = pywt.wavedec(arr_clean, wavelet, level=actual_level)
            total_energy = sum(np.sum(c ** 2) for c in coeffs)

            for i, coeff in enumerate(coeffs):
                level_energy = np.sum(coeff ** 2)
                normalized = level_energy / total_energy if total_energy > 1e-12 else 0.0
                features[f"wavelet_energy_level{i}"] = float(normalized)

        except ImportError:
            logger.warning("pywt không khả dụng, bỏ qua wavelet features")
            for i in range(levels + 1):
                features[f"wavelet_energy_level{i}"] = 0.0

        return features

    # ------------------------------------------------------------------
    # Statistical features
    # ------------------------------------------------------------------

    def extract_statistical_features(
        self,
        data: Union[np.ndarray, pd.Series],
    ) -> Dict[str, float]:
        """Trích xuất đặc trưng thống kê.

        Args:
            data: Tín hiệu đầu vào.

        Returns:
            Dict các đặc trưng: percentiles, variance, coefficient_of_variation.
        """
        arr = np.asarray(data, dtype=float)
        arr_clean = arr[~np.isnan(arr)]
        if len(arr_clean) == 0:
            return {"p25": 0.0, "p50": 0.0, "p75": 0.0, "variance": 0.0, "cv": 0.0}

        mean = np.mean(arr_clean)
        std = np.std(arr_clean)

        return {
            "p25": float(np.percentile(arr_clean, 25)),
            "p50": float(np.percentile(arr_clean, 50)),
            "p75": float(np.percentile(arr_clean, 75)),
            "variance": float(np.var(arr_clean)),
            "cv": float(std / mean) if abs(mean) > 1e-12 else 0.0,
        }

    # ------------------------------------------------------------------
    # Rolling features
    # ------------------------------------------------------------------

    def extract_rolling_features(
        self,
        data: Union[np.ndarray, pd.Series],
        windows: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """Trích xuất rolling features.

        Args:
            data: Tín hiệu đầu vào.
            windows: Danh sách kích thước window. Dùng self.rolling_windows nếu None.

        Returns:
            Dict các đặc trưng rolling cho từng window size.
        """
        windows = windows or self.rolling_windows
        series = pd.Series(np.asarray(data, dtype=float))
        features: Dict[str, float] = {}

        for w in windows:
            rolled = series.rolling(window=w, min_periods=1)
            features[f"rolling_mean_w{w}"] = float(rolled.mean().iloc[-1])
            features[f"rolling_std_w{w}"] = float(rolled.std().iloc[-1])
            features[f"rolling_min_w{w}"] = float(rolled.min().iloc[-1])
            features[f"rolling_max_w{w}"] = float(rolled.max().iloc[-1])

        return features

    # ------------------------------------------------------------------
    # Cross features (domain-specific)
    # ------------------------------------------------------------------

    def extract_cross_features(
        self,
        df: pd.DataFrame,
        power_col: str = "LV ActivePower (kW)",
        wind_speed_col: str = "Wind Speed (m/s)",
    ) -> Dict[str, float]:
        """Trích xuất đặc trưng kết hợp giữa các tín hiệu.

        Args:
            df: DataFrame chứa dữ liệu SCADA.
            power_col: Tên cột công suất.
            wind_speed_col: Tên cột tốc độ gió.

        Returns:
            Dict với power_coefficient, capacity_factor.
        """
        features: Dict[str, float] = {}

        if power_col in df.columns and wind_speed_col in df.columns:
            power = df[power_col].mean()
            wind_speed = df[wind_speed_col].mean()

            # Power coefficient: Cp = P / (0.5 * rho * A * V^3)
            # Đơn giản hóa: sử dụng tỉ lệ công suất / gió^3
            if abs(wind_speed) > 0.1:
                features["power_coefficient"] = float(power / (wind_speed ** 3))
            else:
                features["power_coefficient"] = 0.0

            # Capacity factor: tỉ lệ giữa công suất thực và công suất lý thuyết
            if "Theoretical_Power_Curve (KWh)" in df.columns:
                theoretical = df["Theoretical_Power_Curve (KWh)"].mean()
                features["capacity_factor"] = float(power / theoretical) if theoretical > 1e-6 else 0.0
            else:
                features["capacity_factor"] = 0.0

        return features

    # ------------------------------------------------------------------
    # Feature matrix
    # ------------------------------------------------------------------

    def create_feature_matrix(
        self,
        df: pd.DataFrame,
        window_size: Optional[int] = None,
        numeric_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Tạo ma trận đặc trưng từ sliding window.

        Args:
            df: DataFrame với dữ liệu SCADA.
            window_size: Kích thước window. Dùng self.window_size nếu None.
            numeric_cols: Danh sách cột tín hiệu. None = tất cả cột số.

        Returns:
            DataFrame ma trận đặc trưng.
        """
        window_size = window_size or self.window_size
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        all_features = []
        for i in range(window_size, len(df) + 1, window_size):
            window = df.iloc[max(0, i - window_size):i]
            row: Dict[str, float] = {}

            for col in numeric_cols:
                if col not in window.columns:
                    continue
                col_data = window[col].values
                prefix = col.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")

                td = self.extract_time_domain_features(col_data)
                row.update({f"{prefix}_{k}": v for k, v in td.items()})

                fd = self.extract_frequency_domain_features(col_data)
                row.update({f"{prefix}_{k}": v for k, v in fd.items()})

                sd = self.extract_statistical_features(col_data)
                row.update({f"{prefix}_{k}": v for k, v in sd.items()})

                wf = self.extract_wavelet_features(col_data)
                row.update({f"{prefix}_{k}": v for k, v in wf.items()})

            # Cross features
            cf = self.extract_cross_features(window)
            row.update(cf)

            if isinstance(df.index, pd.DatetimeIndex):
                row["timestamp"] = df.index[min(i - 1, len(df) - 1)]

            all_features.append(row)

        feature_df = pd.DataFrame(all_features)
        logger.info(
            "Feature matrix: %d samples x %d features",
            len(feature_df), len(feature_df.columns),
        )
        return feature_df

    # ------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------

    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = "rfe",
        n_features: int = 20,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Lựa chọn đặc trưng quan trọng.

        Args:
            X: Ma trận đặc trưng.
            y: Nhãn.
            method: Phương pháp: 'rfe' (Recursive Feature Elimination).
            n_features: Số đặc trưng cần chọn.

        Returns:
            Tuple (X_selected, selected_feature_names).
        """
        n_features = min(n_features, X.shape[1])

        if method == "rfe":
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            selector = RFE(estimator, n_features_to_select=n_features, step=1)
            selector.fit(X.fillna(0), y)
            selected_cols = X.columns[selector.support_].tolist()
        else:
            # Fallback: lấy n_features đầu tiên
            selected_cols = X.columns[:n_features].tolist()

        logger.info("Feature selection (%s): %d features", method, len(selected_cols))
        return X[selected_cols], selected_cols

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def normalize_features(
        self,
        X: pd.DataFrame,
        method: str = "standard",
        fit: bool = True,
    ) -> pd.DataFrame:
        """Chuẩn hóa đặc trưng.

        Args:
            X: Ma trận đặc trưng.
            method: 'standard', 'minmax', hoặc 'robust'.
            fit: True để fit scaler mới, False để dùng scaler đã fit.

        Returns:
            DataFrame đã chuẩn hóa.
        """
        if method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()

        if fit:
            self._scaler = scaler
            X_scaled = scaler.fit_transform(X.fillna(0))
        else:
            if self._scaler is None:
                logger.warning("Scaler chưa được fit. Tự động fit mới.")
                self._scaler = scaler
                X_scaled = scaler.fit_transform(X.fillna(0))
            else:
                X_scaled = self._scaler.transform(X.fillna(0))

        logger.info("Normalize (%s): %d samples x %d features", method, *X.shape)
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
