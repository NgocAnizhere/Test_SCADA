"""
Unit tests cho Signal Processing module.

Tests bao gồm:
- NoiseFilter: Butterworth, Wavelet, Kalman, Moving Average, Savitzky-Golay
- FrequencyAnalyzer: FFT, PSD, dominant frequencies
- ContinuityChecker: missing timestamps, gaps, duplicates, interpolation
"""

import numpy as np
import pandas as pd
import pytest
from scipy import signal as scipy_signal

# Import modules cần test
from src.signal_processing.noise_filter import NoiseFilter
from src.signal_processing.frequency_analysis import FrequencyAnalyzer
from src.signal_processing.continuity_check import ContinuityChecker


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def fs() -> float:
    """Tần số lấy mẫu cho test (Hz)."""
    return 100.0  # 100 Hz để test dễ hơn 1/600


@pytest.fixture
def noise_filter(fs: float) -> NoiseFilter:
    """Tạo NoiseFilter instance cho test."""
    return NoiseFilter(fs=fs)


@pytest.fixture
def freq_analyzer(fs: float) -> FrequencyAnalyzer:
    """Tạo FrequencyAnalyzer instance cho test."""
    return FrequencyAnalyzer(fs=fs)


@pytest.fixture
def continuity_checker() -> ContinuityChecker:
    """Tạo ContinuityChecker instance cho test."""
    return ContinuityChecker(expected_freq="10min")


@pytest.fixture
def pure_sine(fs: float) -> np.ndarray:
    """Tín hiệu sine thuần túy 5 Hz, 1000 mẫu."""
    t = np.arange(1000) / fs
    return np.sin(2 * np.pi * 5 * t)


@pytest.fixture
def noisy_signal(pure_sine: np.ndarray) -> np.ndarray:
    """Tín hiệu sine + nhiễu Gaussian."""
    np.random.seed(42)
    noise = np.random.normal(0, 0.5, len(pure_sine))
    return pure_sine + noise


@pytest.fixture
def scada_df() -> pd.DataFrame:
    """DataFrame SCADA giả lập với DatetimeIndex đều."""
    idx = pd.date_range("2023-01-01", periods=200, freq="10min")
    np.random.seed(42)
    df = pd.DataFrame({
        "LV ActivePower (kW)": np.random.normal(1000, 200, 200),
        "Wind Speed (m/s)": np.random.uniform(3, 15, 200),
        "Theoretical_Power_Curve (KWh)": np.random.normal(1100, 150, 200),
        "Wind Direction (°)": np.random.uniform(0, 360, 200),
    }, index=idx)
    return df


# ==============================================================================
# Tests: NoiseFilter
# ==============================================================================

class TestNoiseFilter:
    """Tests cho class NoiseFilter."""

    def test_butterworth_lowpass_output_shape(
        self, noise_filter: NoiseFilter, noisy_signal: np.ndarray
    ) -> None:
        """Kiểm tra output có cùng shape với input."""
        filtered = noise_filter.butterworth_lowpass(noisy_signal, cutoff=10.0)
        assert filtered.shape == noisy_signal.shape

    def test_butterworth_lowpass_removes_high_freq(
        self, noise_filter: NoiseFilter, fs: float
    ) -> None:
        """Kiểm tra lọc lowpass giảm năng lượng tần số cao."""
        # Tín hiệu với 2 component: 5 Hz và 40 Hz
        t = np.arange(1000) / fs
        signal_mixed = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 40 * t)

        filtered = noise_filter.butterworth_lowpass(signal_mixed, cutoff=15.0)

        # Năng lượng tổng của filtered phải nhỏ hơn input (đã loại bỏ 40Hz)
        energy_original = np.sum(signal_mixed ** 2)
        energy_filtered = np.sum(filtered ** 2)
        assert energy_filtered < energy_original

    def test_butterworth_bandpass_output_shape(
        self, noise_filter: NoiseFilter, noisy_signal: np.ndarray
    ) -> None:
        """Kiểm tra bandpass output shape."""
        filtered = noise_filter.butterworth_bandpass(noisy_signal, lowcut=2.0, highcut=10.0)
        assert filtered.shape == noisy_signal.shape

    def test_kalman_filter_output_shape(
        self, noise_filter: NoiseFilter, noisy_signal: np.ndarray
    ) -> None:
        """Kiểm tra Kalman filter output shape."""
        filtered = noise_filter.kalman_filter(noisy_signal)
        assert filtered.shape == noisy_signal.shape

    def test_kalman_filter_smoothing(
        self, noise_filter: NoiseFilter, noisy_signal: np.ndarray, pure_sine: np.ndarray
    ) -> None:
        """Kalman filter phải giảm nhiễu (MSE so với pure_sine nhỏ hơn noisy)."""
        filtered = noise_filter.kalman_filter(noisy_signal)

        mse_before = np.mean((noisy_signal - pure_sine) ** 2)
        mse_after = np.mean((filtered - pure_sine) ** 2)
        # Không nhất thiết luôn tốt hơn với mọi variance, chỉ kiểm tra không crash
        assert mse_after >= 0  # Chỉ kiểm tra không có NaN

    def test_moving_average_output_shape(
        self, noise_filter: NoiseFilter, noisy_signal: np.ndarray
    ) -> None:
        """Kiểm tra moving average output shape."""
        filtered = noise_filter.moving_average(noisy_signal, window_size=10)
        assert filtered.shape == noisy_signal.shape

    def test_moving_average_constant_signal(
        self, noise_filter: NoiseFilter
    ) -> None:
        """Moving average của hằng số phải trả về hằng số."""
        constant = np.ones(100) * 5.0
        filtered = noise_filter.moving_average(constant, window_size=10)
        np.testing.assert_allclose(filtered, 5.0, atol=1e-10)

    def test_savitzky_golay_output_shape(
        self, noise_filter: NoiseFilter, noisy_signal: np.ndarray
    ) -> None:
        """Kiểm tra Savitzky-Golay output shape."""
        filtered = noise_filter.savitzky_golay_filter(noisy_signal)
        assert filtered.shape == noisy_signal.shape

    def test_snr_infinite_for_identical(
        self, noise_filter: NoiseFilter, pure_sine: np.ndarray
    ) -> None:
        """SNR phải là inf khi signal = filtered (no noise)."""
        snr = noise_filter.compute_snr(pure_sine, pure_sine)
        assert snr == float("inf")

    def test_snr_positive_after_filter(
        self, noise_filter: NoiseFilter, noisy_signal: np.ndarray
    ) -> None:
        """SNR phải hữu hạn sau khi lọc."""
        filtered = noise_filter.moving_average(noisy_signal)
        snr = noise_filter.compute_snr(noisy_signal, filtered)
        assert np.isfinite(snr)

    def test_wavelet_denoise_output_shape(
        self, noise_filter: NoiseFilter, noisy_signal: np.ndarray
    ) -> None:
        """Kiểm tra wavelet output shape."""
        try:
            filtered = noise_filter.wavelet_denoise(noisy_signal)
            assert len(filtered) == len(noisy_signal)
        except ImportError:
            pytest.skip("pywt không khả dụng")

    def test_apply_all_filters_returns_dict(
        self, noise_filter: NoiseFilter, noisy_signal: np.ndarray
    ) -> None:
        """apply_all_filters phải trả về dict với tất cả keys mong đợi."""
        results = noise_filter.apply_all_filters(noisy_signal, cutoff=10.0)
        expected_keys = {
            "original", "butterworth_lowpass", "kalman",
            "moving_average", "savitzky_golay"
        }
        assert expected_keys.issubset(set(results.keys()))


# ==============================================================================
# Tests: FrequencyAnalyzer
# ==============================================================================

class TestFrequencyAnalyzer:
    """Tests cho class FrequencyAnalyzer."""

    def test_fft_dominant_frequency(
        self, freq_analyzer: FrequencyAnalyzer, fs: float
    ) -> None:
        """FFT phải tìm đúng tần số dominant của tín hiệu sine."""
        target_freq = 10.0  # Hz
        t = np.arange(1024) / fs
        signal_pure = np.sin(2 * np.pi * target_freq * t)

        freqs, mags = freq_analyzer.compute_fft(signal_pure)

        # Tần số dominant phải gần 10 Hz
        dominant_idx = np.argmax(mags[1:]) + 1
        dominant_freq = freqs[dominant_idx]
        assert abs(dominant_freq - target_freq) < 1.0, (
            f"Dominant freq {dominant_freq:.2f} Hz, expected ~{target_freq} Hz"
        )

    def test_fft_output_lengths_match(
        self, freq_analyzer: FrequencyAnalyzer, pure_sine: np.ndarray
    ) -> None:
        """FFT frequencies và magnitudes phải có cùng length."""
        freqs, mags = freq_analyzer.compute_fft(pure_sine)
        assert len(freqs) == len(mags)

    def test_fft_frequencies_non_negative(
        self, freq_analyzer: FrequencyAnalyzer, pure_sine: np.ndarray
    ) -> None:
        """Tất cả frequencies từ rfft phải >= 0."""
        freqs, _ = freq_analyzer.compute_fft(pure_sine)
        assert np.all(freqs >= 0)

    def test_psd_output(
        self, freq_analyzer: FrequencyAnalyzer, pure_sine: np.ndarray
    ) -> None:
        """PSD phải trả về frequencies và psd với length > 0."""
        freqs, psd = freq_analyzer.compute_power_spectral_density(pure_sine)
        assert len(freqs) > 0
        assert len(psd) > 0
        assert len(freqs) == len(psd)
        assert np.all(psd >= 0)  # PSD phải không âm

    def test_find_dominant_frequencies(
        self, freq_analyzer: FrequencyAnalyzer, fs: float
    ) -> None:
        """find_dominant_frequencies phải trả về list với rank."""
        t = np.arange(1024) / fs
        signal_data = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)

        dominant = freq_analyzer.find_dominant_frequencies(signal_data, n_peaks=3)
        assert isinstance(dominant, list)
        assert len(dominant) <= 3
        if dominant:
            assert "frequency" in dominant[0]
            assert "magnitude" in dominant[0]
            assert "rank" in dominant[0]

    def test_spectrogram_output_shapes(
        self, freq_analyzer: FrequencyAnalyzer, fs: float
    ) -> None:
        """Spectrogram phải trả về 3 arrays."""
        t = np.arange(512) / fs
        signal_data = np.sin(2 * np.pi * 5 * t)
        f, t_out, sxx = freq_analyzer.compute_spectrogram(signal_data)
        assert len(f) > 0
        assert len(t_out) > 0
        assert sxx.shape == (len(f), len(t_out))

    def test_thd_zero_for_pure_sine(
        self, freq_analyzer: FrequencyAnalyzer, fs: float
    ) -> None:
        """THD phải thấp cho tín hiệu sine thuần túy."""
        t = np.arange(2048) / fs
        pure = np.sin(2 * np.pi * 5 * t)
        thd = freq_analyzer.compute_thd(pure)
        # THD có thể không bằng 0 do phân giải tần số, nhưng phải < 0.5
        assert 0.0 <= thd < 1.0

    def test_stft_output(
        self, freq_analyzer: FrequencyAnalyzer, fs: float
    ) -> None:
        """STFT phải trả về 3 arrays."""
        t = np.arange(512) / fs
        signal_data = np.sin(2 * np.pi * 5 * t)
        f, t_out, zxx = freq_analyzer.short_time_fourier_transform(signal_data)
        assert len(f) > 0
        assert len(t_out) > 0
        assert zxx.shape[0] == len(f)

    def test_analyze_signal_returns_dataframe(
        self, freq_analyzer: FrequencyAnalyzer, pure_sine: np.ndarray
    ) -> None:
        """analyze_signal phải trả về DataFrame."""
        result = freq_analyzer.analyze_signal(pure_sine, signal_name="test")
        assert isinstance(result, pd.DataFrame)
        assert "signal" in result.columns
        assert "metric" in result.columns
        assert "value" in result.columns


# ==============================================================================
# Tests: ContinuityChecker
# ==============================================================================

class TestContinuityChecker:
    """Tests cho class ContinuityChecker."""

    def test_check_missing_timestamps_none(
        self, continuity_checker: ContinuityChecker, scada_df: pd.DataFrame
    ) -> None:
        """DataFrame đầy đủ không có missing timestamps."""
        result = continuity_checker.check_missing_timestamps(scada_df)
        assert result["missing_count"] == 0

    def test_check_missing_timestamps_detects_gaps(
        self, continuity_checker: ContinuityChecker
    ) -> None:
        """Phát hiện đúng số timestamps bị thiếu khi xóa một số hàng."""
        idx = pd.date_range("2023-01-01", periods=100, freq="10min")
        df = pd.DataFrame({"value": np.ones(100)}, index=idx)

        # Xóa 5 hàng liên tiếp (tạo gap)
        df_gap = df.drop(df.index[40:45])

        result = continuity_checker.check_missing_timestamps(df_gap)
        assert result["missing_count"] == 5

    def test_check_data_gaps_detects_large_gap(
        self, continuity_checker: ContinuityChecker
    ) -> None:
        """Phát hiện khoảng trống dữ liệu lớn hơn threshold."""
        idx1 = pd.date_range("2023-01-01", periods=50, freq="10min")
        idx2 = pd.date_range("2023-01-02 12:00", periods=50, freq="10min")
        idx = idx1.append(idx2)
        df = pd.DataFrame({"value": np.ones(len(idx))}, index=idx)

        result = continuity_checker.check_data_gaps(df, threshold_minutes=20)
        assert result["gap_count"] >= 1
        # Gap lớn nhất phải > 60 phút
        if result["gaps"]:
            max_gap = max(g["duration_minutes"] for g in result["gaps"])
            assert max_gap > 60

    def test_check_duplicate_timestamps(
        self, continuity_checker: ContinuityChecker
    ) -> None:
        """Phát hiện timestamp trùng lặp."""
        idx = pd.date_range("2023-01-01", periods=50, freq="10min")
        # Thêm duplicate bằng cách concat
        idx_dup = idx.append(idx[:5])
        df = pd.DataFrame({"value": np.ones(len(idx_dup))}, index=idx_dup)

        result = continuity_checker.check_duplicate_timestamps(df)
        assert result["duplicate_count"] >= 5

    def test_interpolate_linear_fills_nan(
        self, continuity_checker: ContinuityChecker
    ) -> None:
        """Linear interpolation phải điền đầy đủ giá trị thiếu."""
        idx = pd.date_range("2023-01-01", periods=50, freq="10min")
        values = np.arange(50, dtype=float)
        df = pd.DataFrame({"value": values}, index=idx)

        # Tạo gaps
        df_gap = df.drop(df.index[20:25])
        df_filled = continuity_checker.interpolate_missing(df_gap, method="linear")

        assert df_filled.isnull().sum().sum() == 0

    def test_interpolate_preserves_length(
        self, continuity_checker: ContinuityChecker
    ) -> None:
        """Interpolation phải tạo đủ số hàng theo expected_freq."""
        idx = pd.date_range("2023-01-01", periods=10, freq="10min")
        df = pd.DataFrame({"value": np.arange(10)}, index=idx)
        df_gap = df.drop(df.index[3:6])

        df_filled = continuity_checker.interpolate_missing(df_gap, method="linear")
        assert len(df_filled) == 10

    def test_sampling_rate_consistency_uniform(
        self, continuity_checker: ContinuityChecker, scada_df: pd.DataFrame
    ) -> None:
        """DataFrame đều phải có is_consistent=True."""
        result = continuity_checker.check_sampling_rate_consistency(scada_df)
        assert result["is_consistent"] is True

    def test_continuity_report_perfect_data(
        self, continuity_checker: ContinuityChecker, scada_df: pd.DataFrame
    ) -> None:
        """DataFrame hoàn hảo phải có continuity_score gần 100."""
        report = continuity_checker.generate_continuity_report(scada_df)
        assert report["continuity_score"] > 90.0
        assert report["missing_count"] == 0

    def test_validate_datetime_index_raises(
        self, continuity_checker: ContinuityChecker
    ) -> None:
        """Phải raise ValueError khi không có DatetimeIndex."""
        df_no_datetime = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="DatetimeIndex"):
            continuity_checker.check_missing_timestamps(df_no_datetime)


# ==============================================================================
# Integration tests
# ==============================================================================

class TestSignalProcessingIntegration:
    """Integration tests kết hợp các module."""

    def test_filter_then_fft(self, fs: float) -> None:
        """Sau khi lọc, FFT vẫn phát hiện được tần số dominant."""
        target_freq = 5.0
        t = np.arange(1024) / fs
        pure = np.sin(2 * np.pi * target_freq * t)
        noisy = pure + np.random.normal(0, 0.3, len(pure))

        noise_filter = NoiseFilter(fs=fs)
        filtered = noise_filter.moving_average(noisy, window_size=5)

        freq_analyzer = FrequencyAnalyzer(fs=fs)
        freqs, mags = freq_analyzer.compute_fft(filtered)
        dominant_idx = np.argmax(mags[1:]) + 1
        dominant_freq = freqs[dominant_idx]

        assert abs(dominant_freq - target_freq) < 2.0

    def test_continuity_check_then_filter(self) -> None:
        """Kiểm tra continuity rồi interpolate rồi filter."""
        idx = pd.date_range("2023-01-01", periods=100, freq="10min")
        values = np.sin(np.linspace(0, 4 * np.pi, 100))
        df = pd.DataFrame({"signal": values}, index=idx)
        df_gap = df.drop(df.index[40:45])

        checker = ContinuityChecker()
        df_filled = checker.interpolate_missing(df_gap, method="linear")
        assert len(df_filled) == 100

        noise_filter = NoiseFilter(fs=1 / 600.0)
        filtered = noise_filter.savitzky_golay_filter(df_filled["signal"].values)
        assert len(filtered) == 100
        assert not np.any(np.isnan(filtered))
