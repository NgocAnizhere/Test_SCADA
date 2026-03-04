"""
Frequency Analyzer - phân tích tần số tín hiệu SCADA.

Cung cấp FFT, PSD (Welch), Spectrogram, STFT, THD và phát hiện
tần số bất thường.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks, welch, stft, spectrogram

logger = logging.getLogger(__name__)


class FrequencyAnalyzer:
    """Phân tích tần số tín hiệu SCADA.

    Attributes:
        fs (float): Tần số lấy mẫu (Hz).
    """

    def __init__(self, fs: float = 1 / 600.0) -> None:
        """Khởi tạo FrequencyAnalyzer.

        Args:
            fs: Tần số lấy mẫu (Hz). Mặc định 1/600 (10 phút).
        """
        self.fs = fs
        logger.debug("FrequencyAnalyzer khởi tạo, fs=%.6f Hz", fs)

    # ------------------------------------------------------------------
    # FFT
    # ------------------------------------------------------------------

    def compute_fft(
        self,
        signal_data: Union[np.ndarray, pd.Series],
        fs: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Tính FFT của tín hiệu.

        Args:
            signal_data: Tín hiệu đầu vào.
            fs: Tần số lấy mẫu. Dùng self.fs nếu None.

        Returns:
            Tuple (frequencies, magnitudes) - tần số và biên độ.
        """
        fs = fs or self.fs
        data = np.asarray(signal_data, dtype=float)
        # Xử lý NaN
        data = pd.Series(data).ffill().bfill().values

        n = len(data)
        fft_vals = np.fft.rfft(data)
        frequencies = np.fft.rfftfreq(n, d=1.0 / fs)
        magnitudes = np.abs(fft_vals) / n
        logger.debug("FFT: n=%d, freq_resolution=%.6f Hz", n, frequencies[1] if len(frequencies) > 1 else 0)
        return frequencies, magnitudes

    # ------------------------------------------------------------------
    # Power Spectral Density
    # ------------------------------------------------------------------

    def compute_power_spectral_density(
        self,
        signal_data: Union[np.ndarray, pd.Series],
        fs: Optional[float] = None,
        nperseg: int = 256,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Tính PSD bằng phương pháp Welch.

        Args:
            signal_data: Tín hiệu đầu vào.
            fs: Tần số lấy mẫu. Dùng self.fs nếu None.
            nperseg: Số điểm mỗi segment.

        Returns:
            Tuple (frequencies, psd) - tần số và mật độ phổ công suất.
        """
        fs = fs or self.fs
        data = np.asarray(signal_data, dtype=float)
        data = pd.Series(data).ffill().bfill().values

        # Điều chỉnh nperseg nếu dữ liệu ngắn
        nperseg = min(nperseg, len(data))
        frequencies, psd = welch(data, fs=fs, nperseg=nperseg)
        logger.debug("PSD (Welch): nperseg=%d, n_freq=%d", nperseg, len(frequencies))
        return frequencies, psd

    # ------------------------------------------------------------------
    # Spectrogram
    # ------------------------------------------------------------------

    def compute_spectrogram(
        self,
        signal_data: Union[np.ndarray, pd.Series],
        fs: Optional[float] = None,
        nperseg: int = 256,
        noverlap: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tính Spectrogram của tín hiệu.

        Args:
            signal_data: Tín hiệu đầu vào.
            fs: Tần số lấy mẫu. Dùng self.fs nếu None.
            nperseg: Số điểm mỗi segment.
            noverlap: Số điểm overlap.

        Returns:
            Tuple (frequencies, times, Sxx) - tần số, thời gian, spectrogram.
        """
        fs = fs or self.fs
        data = np.asarray(signal_data, dtype=float)
        data = pd.Series(data).ffill().bfill().values

        nperseg = min(nperseg, len(data))
        noverlap = noverlap if noverlap is not None else nperseg // 2

        frequencies, times, sxx = spectrogram(
            data, fs=fs, nperseg=nperseg, noverlap=noverlap
        )
        logger.debug("Spectrogram: freq=%d, time=%d", len(frequencies), len(times))
        return frequencies, times, sxx

    # ------------------------------------------------------------------
    # Dominant frequencies
    # ------------------------------------------------------------------

    def find_dominant_frequencies(
        self,
        signal_data: Union[np.ndarray, pd.Series],
        fs: Optional[float] = None,
        n_peaks: int = 5,
    ) -> List[Dict]:
        """Tìm n tần số dominant trong phổ FFT.

        Args:
            signal_data: Tín hiệu đầu vào.
            fs: Tần số lấy mẫu. Dùng self.fs nếu None.
            n_peaks: Số tần số dominant cần tìm.

        Returns:
            List các dict với 'frequency', 'magnitude', 'rank'.
        """
        frequencies, magnitudes = self.compute_fft(signal_data, fs)

        # Tìm peaks
        peaks, properties = find_peaks(magnitudes, height=np.max(magnitudes) * 0.01)
        if len(peaks) == 0:
            return []

        # Sắp xếp theo biên độ giảm dần
        sorted_idx = np.argsort(magnitudes[peaks])[::-1]
        top_peaks = sorted_idx[:n_peaks]

        result = []
        for rank, idx in enumerate(top_peaks, 1):
            peak_idx = peaks[idx]
            result.append({
                "rank": rank,
                "frequency": float(frequencies[peak_idx]),
                "magnitude": float(magnitudes[peak_idx]),
            })

        logger.debug("Tìm thấy %d tần số dominant", len(result))
        return result

    # ------------------------------------------------------------------
    # Total Harmonic Distortion
    # ------------------------------------------------------------------

    def compute_thd(
        self,
        signal_data: Union[np.ndarray, pd.Series],
        fs: Optional[float] = None,
        n_harmonics: int = 5,
    ) -> float:
        """Tính Total Harmonic Distortion (THD).

        Args:
            signal_data: Tín hiệu đầu vào.
            fs: Tần số lấy mẫu. Dùng self.fs nếu None.
            n_harmonics: Số harmonic cần xét.

        Returns:
            THD (tỉ lệ, không phải phần trăm).
        """
        frequencies, magnitudes = self.compute_fft(signal_data, fs)

        if len(magnitudes) < 2:
            return 0.0

        # Tần số cơ bản là tần số có biên độ lớn nhất (bỏ DC)
        fundamental_idx = np.argmax(magnitudes[1:]) + 1
        fundamental_mag = magnitudes[fundamental_idx]
        fundamental_freq = frequencies[fundamental_idx]

        if fundamental_mag < 1e-12:
            return 0.0

        # Tính tổng công suất harmonic
        harmonic_power = 0.0
        freq_resolution = frequencies[1] - frequencies[0] if len(frequencies) > 1 else 1.0

        for h in range(2, n_harmonics + 2):
            harmonic_freq = h * fundamental_freq
            # Tìm index gần nhất với harmonic_freq
            harmonic_idx = int(round(harmonic_freq / freq_resolution)) if freq_resolution > 0 else 0
            if 0 < harmonic_idx < len(magnitudes):
                harmonic_power += magnitudes[harmonic_idx] ** 2

        thd = np.sqrt(harmonic_power) / fundamental_mag
        logger.debug(
            "THD: fundamental=%.4f Hz, thd=%.4f", fundamental_freq, thd
        )
        return float(thd)

    # ------------------------------------------------------------------
    # STFT
    # ------------------------------------------------------------------

    def short_time_fourier_transform(
        self,
        signal_data: Union[np.ndarray, pd.Series],
        fs: Optional[float] = None,
        nperseg: int = 256,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tính Short-Time Fourier Transform (STFT).

        Args:
            signal_data: Tín hiệu đầu vào.
            fs: Tần số lấy mẫu. Dùng self.fs nếu None.
            nperseg: Số điểm mỗi segment.

        Returns:
            Tuple (frequencies, times, Zxx) - tần số, thời gian, STFT phức.
        """
        fs = fs or self.fs
        data = np.asarray(signal_data, dtype=float)
        data = pd.Series(data).ffill().bfill().values

        nperseg = min(nperseg, len(data))
        f, t, zxx = stft(data, fs=fs, nperseg=nperseg)
        logger.debug("STFT: freq=%d, time=%d", len(f), len(t))
        return f, t, zxx

    # ------------------------------------------------------------------
    # Frequency anomaly detection
    # ------------------------------------------------------------------

    def detect_frequency_anomalies(
        self,
        signal_data: Union[np.ndarray, pd.Series],
        fs: Optional[float] = None,
        normal_band: Tuple[float, float] = (0.0, 0.01),
    ) -> Dict:
        """Phát hiện tần số bất thường ngoài dải hoạt động bình thường.

        Args:
            signal_data: Tín hiệu đầu vào.
            fs: Tần số lấy mẫu. Dùng self.fs nếu None.
            normal_band: Tuple (low, high) dải tần bình thường (Hz).

        Returns:
            Dict với 'anomalous_frequencies', 'anomalous_power_ratio', 'is_anomalous'.
        """
        frequencies, magnitudes = self.compute_fft(signal_data, fs)

        total_power = np.sum(magnitudes ** 2)
        if total_power < 1e-12:
            return {
                "anomalous_frequencies": [],
                "anomalous_power_ratio": 0.0,
                "is_anomalous": False,
            }

        # Tính công suất ngoài dải bình thường
        outside_band = (frequencies < normal_band[0]) | (frequencies > normal_band[1])
        anomalous_power = np.sum(magnitudes[outside_band] ** 2)
        anomalous_ratio = anomalous_power / total_power

        # Tìm các tần số có biên độ cao ngoài dải
        threshold = np.max(magnitudes) * 0.1
        anomalous_mask = outside_band & (magnitudes > threshold)
        anomalous_freqs = frequencies[anomalous_mask].tolist()

        result = {
            "anomalous_frequencies": anomalous_freqs,
            "anomalous_power_ratio": float(anomalous_ratio),
            "is_anomalous": anomalous_ratio > 0.5,
        }
        logger.debug(
            "Frequency anomaly: ratio=%.4f, n_anomalous=%d",
            anomalous_ratio, len(anomalous_freqs),
        )
        return result

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def analyze_signal(
        self,
        signal_data: Union[np.ndarray, pd.Series],
        fs: Optional[float] = None,
        signal_name: str = "signal",
    ) -> pd.DataFrame:
        """Phân tích tổng hợp tín hiệu và trả về DataFrame.

        Args:
            signal_data: Tín hiệu đầu vào.
            fs: Tần số lấy mẫu. Dùng self.fs nếu None.
            signal_name: Tên tín hiệu (dùng làm nhãn).

        Returns:
            DataFrame tổng hợp kết quả phân tích.
        """
        dominant = self.find_dominant_frequencies(signal_data, fs, n_peaks=3)
        thd = self.compute_thd(signal_data, fs)
        freq_anomaly = self.detect_frequency_anomalies(signal_data, fs)

        rows = []
        for peak in dominant:
            rows.append({
                "signal": signal_name,
                "metric": f"dominant_freq_rank{peak['rank']}",
                "value": peak["frequency"],
                "unit": "Hz",
            })

        rows.append({"signal": signal_name, "metric": "thd", "value": thd, "unit": ""})
        rows.append({
            "signal": signal_name,
            "metric": "anomalous_power_ratio",
            "value": freq_anomaly["anomalous_power_ratio"],
            "unit": "",
        })

        return pd.DataFrame(rows)
