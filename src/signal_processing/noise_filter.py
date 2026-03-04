"""
Noise Filter - các bộ lọc tín hiệu cho dữ liệu SCADA.

Module cung cấp nhiều phương pháp lọc nhiễu:
Butterworth, Wavelet, Kalman, Moving Average, Savitzky-Golay.
"""

import logging
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt, savgol_filter

logger = logging.getLogger(__name__)


class NoiseFilter:
    """Bộ lọc nhiễu tín hiệu SCADA.

    Cung cấp các phương pháp lọc nhiễu phổ biến và tính SNR.

    Attributes:
        fs (float): Tần số lấy mẫu (Hz).
    """

    def __init__(self, fs: float = 1 / 600.0) -> None:
        """Khởi tạo NoiseFilter.

        Args:
            fs: Tần số lấy mẫu Hz. Mặc định 1/600 (10 phút).
        """
        self.fs = fs
        logger.debug("NoiseFilter khởi tạo, fs=%.6f Hz", fs)

    # ------------------------------------------------------------------
    # Butterworth filters
    # ------------------------------------------------------------------

    def butterworth_lowpass(
        self,
        data: Union[np.ndarray, pd.Series],
        cutoff: float,
        fs: Optional[float] = None,
        order: int = 4,
    ) -> np.ndarray:
        """Lọc Butterworth low-pass.

        Args:
            data: Tín hiệu đầu vào.
            cutoff: Tần số cắt (Hz).
            fs: Tần số lấy mẫu. Dùng self.fs nếu None.
            order: Bậc bộ lọc.

        Returns:
            Tín hiệu đã lọc dạng numpy array.
        """
        fs = fs or self.fs
        nyq = 0.5 * fs
        normalized_cutoff = cutoff / nyq
        # Đảm bảo normalized_cutoff hợp lệ (0, 1)
        normalized_cutoff = np.clip(normalized_cutoff, 1e-6, 1 - 1e-6)
        b, a = butter(order, normalized_cutoff, btype="low")
        data_array = np.asarray(data, dtype=float)
        filtered = filtfilt(b, a, data_array)
        logger.debug("Butterworth lowpass cutoff=%.4f Hz, order=%d", cutoff, order)
        return filtered

    def butterworth_bandpass(
        self,
        data: Union[np.ndarray, pd.Series],
        lowcut: float,
        highcut: float,
        fs: Optional[float] = None,
        order: int = 4,
    ) -> np.ndarray:
        """Lọc Butterworth band-pass.

        Args:
            data: Tín hiệu đầu vào.
            lowcut: Tần số cắt dưới (Hz).
            highcut: Tần số cắt trên (Hz).
            fs: Tần số lấy mẫu. Dùng self.fs nếu None.
            order: Bậc bộ lọc.

        Returns:
            Tín hiệu đã lọc dạng numpy array.
        """
        fs = fs or self.fs
        nyq = 0.5 * fs
        low = np.clip(lowcut / nyq, 1e-6, 1 - 1e-6)
        high = np.clip(highcut / nyq, 1e-6, 1 - 1e-6)
        b, a = butter(order, [low, high], btype="band")
        data_array = np.asarray(data, dtype=float)
        filtered = filtfilt(b, a, data_array)
        logger.debug(
            "Butterworth bandpass [%.4f, %.4f] Hz, order=%d", lowcut, highcut, order
        )
        return filtered

    # ------------------------------------------------------------------
    # Wavelet filter
    # ------------------------------------------------------------------

    def wavelet_denoise(
        self,
        data: Union[np.ndarray, pd.Series],
        wavelet: str = "db4",
        level: int = 5,
        threshold_method: str = "soft",
    ) -> np.ndarray:
        """Lọc nhiễu bằng Wavelet với soft/hard thresholding.

        Args:
            data: Tín hiệu đầu vào.
            wavelet: Loại wavelet (mặc định 'db4').
            level: Số mức phân tích.
            threshold_method: 'soft' hoặc 'hard'.

        Returns:
            Tín hiệu đã lọc dạng numpy array.
        """
        try:
            import pywt
        except ImportError:
            logger.error("Thư viện pywt chưa cài. Chạy: pip install PyWavelets")
            return np.asarray(data, dtype=float)

        data_array = np.array(data, dtype=float)  # writable copy
        # Thực hiện phân tích wavelet đa mức
        coeffs = pywt.wavedec(data_array, wavelet, level=level)

        # Tính ngưỡng bằng phương pháp VisuShrink
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(data_array)))

        # Áp dụng thresholding trên các hệ số chi tiết (bỏ qua xấp xỉ)
        denoised_coeffs = [coeffs[0]]
        for coeff in coeffs[1:]:
            denoised_coeffs.append(
                pywt.threshold(coeff, threshold, mode=threshold_method)
            )

        # Tái tổng hợp tín hiệu
        denoised = pywt.waverec(denoised_coeffs, wavelet)
        # Đảm bảo cùng độ dài
        denoised = denoised[: len(data_array)]
        logger.debug(
            "Wavelet denoise: wavelet=%s, level=%d, method=%s, threshold=%.4f",
            wavelet, level, threshold_method, threshold,
        )
        return denoised

    # ------------------------------------------------------------------
    # Kalman filter
    # ------------------------------------------------------------------

    def kalman_filter(
        self,
        data: Union[np.ndarray, pd.Series],
        process_variance: float = 1e-5,
        measurement_variance: float = 1e-2,
    ) -> np.ndarray:
        """Bộ lọc Kalman 1D đơn giản.

        Args:
            data: Tín hiệu đầu vào.
            process_variance: Phương sai nhiễu quá trình (Q).
            measurement_variance: Phương sai nhiễu đo lường (R).

        Returns:
            Tín hiệu đã lọc dạng numpy array.
        """
        data_array = np.asarray(data, dtype=float)
        n = len(data_array)
        filtered = np.zeros(n)

        # Khởi tạo
        x_est = data_array[0]   # Ước tính trạng thái
        p_est = 1.0              # Sai số ước tính ban đầu

        for i, measurement in enumerate(data_array):
            # Bước dự đoán
            x_pred = x_est
            p_pred = p_est + process_variance

            # Bước cập nhật
            kalman_gain = p_pred / (p_pred + measurement_variance)
            x_est = x_pred + kalman_gain * (measurement - x_pred)
            p_est = (1 - kalman_gain) * p_pred

            filtered[i] = x_est

        logger.debug(
            "Kalman filter Q=%.2e, R=%.2e", process_variance, measurement_variance
        )
        return filtered

    # ------------------------------------------------------------------
    # Moving average
    # ------------------------------------------------------------------

    def moving_average(
        self,
        data: Union[np.ndarray, pd.Series],
        window_size: int = 10,
    ) -> np.ndarray:
        """Lọc trung bình trượt (Moving Average).

        Args:
            data: Tín hiệu đầu vào.
            window_size: Kích thước cửa sổ trượt.

        Returns:
            Tín hiệu đã lọc dạng numpy array.
        """
        data_array = np.asarray(data, dtype=float)
        kernel = np.ones(window_size) / window_size
        # Dùng 'same' để giữ nguyên độ dài, mode='edge' để xử lý biên
        padded = np.pad(data_array, (window_size // 2, window_size // 2), mode="edge")
        filtered = np.convolve(padded, kernel, mode="valid")
        filtered = filtered[: len(data_array)]
        logger.debug("Moving average window=%d", window_size)
        return filtered

    # ------------------------------------------------------------------
    # Savitzky-Golay filter
    # ------------------------------------------------------------------

    def savitzky_golay_filter(
        self,
        data: Union[np.ndarray, pd.Series],
        window_length: int = 11,
        polyorder: int = 3,
    ) -> np.ndarray:
        """Lọc Savitzky-Golay - làm mịn tín hiệu giữ nguyên đỉnh.

        Args:
            data: Tín hiệu đầu vào.
            window_length: Độ dài cửa sổ (phải lẻ).
            polyorder: Bậc đa thức.

        Returns:
            Tín hiệu đã lọc dạng numpy array.
        """
        data_array = np.asarray(data, dtype=float)
        # Đảm bảo window_length lẻ và >= polyorder+2
        if window_length % 2 == 0:
            window_length += 1
        window_length = max(window_length, polyorder + 2)
        filtered = savgol_filter(data_array, window_length, polyorder)
        logger.debug(
            "Savitzky-Golay window=%d, polyorder=%d", window_length, polyorder
        )
        return filtered

    # ------------------------------------------------------------------
    # Apply all filters
    # ------------------------------------------------------------------

    def apply_all_filters(
        self,
        data: Union[np.ndarray, pd.Series],
        cutoff: float = 0.1,
        wavelet: str = "db4",
        wavelet_level: int = 5,
    ) -> Dict[str, np.ndarray]:
        """Áp dụng tất cả bộ lọc và trả về dict kết quả.

        Args:
            data: Tín hiệu đầu vào.
            cutoff: Tần số cắt cho Butterworth lowpass.
            wavelet: Loại wavelet.
            wavelet_level: Số mức wavelet.

        Returns:
            Dict với key là tên bộ lọc, value là tín hiệu đã lọc.
        """
        data_array = np.asarray(data, dtype=float)
        # Xử lý NaN trước khi lọc
        nan_mask = np.isnan(data_array)
        if nan_mask.any():
            data_clean = pd.Series(data_array).interpolate(method="linear").values
        else:
            data_clean = data_array

        results: Dict[str, np.ndarray] = {
            "original": data_array,
            "butterworth_lowpass": self.butterworth_lowpass(data_clean, cutoff),
            "wavelet": self.wavelet_denoise(data_clean, wavelet, wavelet_level),
            "kalman": self.kalman_filter(data_clean),
            "moving_average": self.moving_average(data_clean),
            "savitzky_golay": self.savitzky_golay_filter(data_clean),
        }

        # Tính SNR cho từng bộ lọc
        snr_results = {}
        for name, filtered in results.items():
            if name != "original":
                snr_before = self._compute_snr(data_clean, data_clean)
                snr_after = self._compute_snr(data_clean, filtered)
                snr_results[name] = {"snr_before": snr_before, "snr_after": snr_after}
                logger.info(
                    "SNR %s: before=%.2f dB, after=%.2f dB", name, snr_before, snr_after
                )

        results["snr"] = snr_results  # type: ignore[assignment]
        return results

    # ------------------------------------------------------------------
    # SNR calculation
    # ------------------------------------------------------------------

    def compute_snr(
        self,
        original: Union[np.ndarray, pd.Series],
        filtered: Union[np.ndarray, pd.Series],
    ) -> float:
        """Tính Signal-to-Noise Ratio (SNR) tính bằng dB.

        Args:
            original: Tín hiệu gốc.
            filtered: Tín hiệu đã lọc (coi là signal).

        Returns:
            SNR tính bằng dB.
        """
        return self._compute_snr(
            np.asarray(original, dtype=float),
            np.asarray(filtered, dtype=float),
        )

    @staticmethod
    def _compute_snr(
        original: np.ndarray,
        filtered: np.ndarray,
    ) -> float:
        """Tính SNR nội bộ.

        Args:
            original: Tín hiệu gốc.
            filtered: Tín hiệu đã lọc.

        Returns:
            SNR tính bằng dB.
        """
        n = min(len(original), len(filtered))
        signal_power = np.mean(filtered[:n] ** 2)
        noise = original[:n] - filtered[:n]
        noise_power = np.mean(noise ** 2)
        if noise_power < 1e-12:
            return float("inf")
        return 10.0 * np.log10(signal_power / noise_power)
