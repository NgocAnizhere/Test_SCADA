"""
Unit tests cho Anomaly Detection module.

Tests bao gồm:
- AnomalyDetector: spike, missing, std_deviation, IQR, drift, flatline
- Isolation Forest và LOF
- classify_anomaly_type
- generate_anomaly_report
"""

import numpy as np
import pandas as pd
import pytest

from src.anomaly_detection.detector import AnomalyDetector


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def detector() -> AnomalyDetector:
    """AnomalyDetector mặc định."""
    return AnomalyDetector(z_threshold=3.0, iqr_multiplier=1.5, missing_threshold=0.05)


@pytest.fixture
def normal_signal() -> np.ndarray:
    """Tín hiệu bình thường (Gaussian)."""
    np.random.seed(42)
    return np.random.normal(100, 5, 500)


@pytest.fixture
def spiked_signal(normal_signal: np.ndarray) -> np.ndarray:
    """Tín hiệu với spikes rõ ràng."""
    s = normal_signal.copy()
    s[100] = 1000.0   # Spike cực lớn
    s[200] = -900.0   # Spike âm
    s[300] = 800.0    # Spike dương
    return s


@pytest.fixture
def signal_with_nan(normal_signal: np.ndarray) -> np.ndarray:
    """Tín hiệu với NaN."""
    s = normal_signal.copy()
    s[50:55] = np.nan
    s[150:153] = np.nan
    return s


@pytest.fixture
def flatline_signal(normal_signal: np.ndarray) -> np.ndarray:
    """Tín hiệu với flatline region."""
    s = normal_signal.copy()
    s[200:260] = 100.0  # Flatline 60 mẫu
    return s


@pytest.fixture
def drift_signal() -> np.ndarray:
    """Tín hiệu với drift step-change mạnh (8% signal ở mức cực cao, z ≈ 3.4)."""
    np.random.seed(42)
    return np.concatenate([
        np.random.normal(0, 0.1, 460),
        np.random.normal(1000, 0.1, 40),
    ])


@pytest.fixture
def scada_df() -> pd.DataFrame:
    """DataFrame SCADA giả lập."""
    np.random.seed(42)
    idx = pd.date_range("2023-01-01", periods=300, freq="10min")
    df = pd.DataFrame({
        "LV ActivePower (kW)": np.random.normal(1000, 200, 300),
        "Wind Speed (m/s)": np.random.uniform(3, 15, 300),
        "Theoretical_Power_Curve (KWh)": np.random.normal(1100, 150, 300),
        "Wind Direction (°)": np.random.uniform(0, 360, 300),
    }, index=idx)
    # Thêm một số anomalies rõ ràng
    df.iloc[100, 0] = 50000.0   # Spike
    df.iloc[200:210, 0] = df.iloc[200, 0]  # Flatline
    return df


# ==============================================================================
# Tests: detect_noise_spikes
# ==============================================================================

class TestDetectNoiseSpikes:
    """Tests cho detect_noise_spikes."""

    def test_detects_known_spikes(
        self, detector: AnomalyDetector, spiked_signal: np.ndarray
    ) -> None:
        """Phải phát hiện được các spike đã biết."""
        anomalies = detector.detect_noise_spikes(spiked_signal)
        assert anomalies[100], "Không phát hiện spike tại index 100"
        assert anomalies[200], "Không phát hiện spike tại index 200"
        assert anomalies[300], "Không phát hiện spike tại index 300"

    def test_no_spikes_in_normal_signal(
        self, detector: AnomalyDetector, normal_signal: np.ndarray
    ) -> None:
        """Tín hiệu bình thường phải có ít spike."""
        anomalies = detector.detect_noise_spikes(normal_signal)
        # Tín hiệu Gaussian với z=3 có ~0.3% anomalies
        anomaly_rate = anomalies.mean()
        assert anomaly_rate < 0.01  # < 1%

    def test_output_is_boolean(
        self, detector: AnomalyDetector, normal_signal: np.ndarray
    ) -> None:
        """Output phải là boolean array."""
        anomalies = detector.detect_noise_spikes(normal_signal)
        assert anomalies.dtype == bool

    def test_output_shape_matches_input(
        self, detector: AnomalyDetector, normal_signal: np.ndarray
    ) -> None:
        """Output shape phải bằng input shape."""
        anomalies = detector.detect_noise_spikes(normal_signal)
        assert anomalies.shape == normal_signal.shape

    def test_constant_signal_no_spikes(self, detector: AnomalyDetector) -> None:
        """Tín hiệu hằng số không có spike."""
        constant = np.ones(100) * 5.0
        anomalies = detector.detect_noise_spikes(constant)
        assert not np.any(anomalies)


# ==============================================================================
# Tests: detect_missing_samples
# ==============================================================================

class TestDetectMissingSamples:
    """Tests cho detect_missing_samples."""

    def test_detects_nan_values(
        self, detector: AnomalyDetector, signal_with_nan: np.ndarray
    ) -> None:
        """Phải phát hiện NaN positions."""
        missing = detector.detect_missing_samples(signal_with_nan)
        # NaN tại [50:55] và [150:153]
        assert np.all(missing[50:55]), "Không phát hiện NaN tại 50:55"
        assert np.all(missing[150:153]), "Không phát hiện NaN tại 150:153"

    def test_no_missing_in_normal(
        self, detector: AnomalyDetector, normal_signal: np.ndarray
    ) -> None:
        """Tín hiệu bình thường không có missing."""
        # Đảm bảo không có 0 trong normal signal
        signal = normal_signal.copy()
        signal[signal == 0] = 0.001
        missing = detector.detect_missing_samples(signal)
        # Ít hoặc không có missing
        assert missing.sum() == 0 or missing.mean() < 0.05

    def test_output_shape(
        self, detector: AnomalyDetector, signal_with_nan: np.ndarray
    ) -> None:
        """Output shape phải bằng input."""
        missing = detector.detect_missing_samples(signal_with_nan)
        assert missing.shape == signal_with_nan.shape


# ==============================================================================
# Tests: detect_flatline
# ==============================================================================

class TestDetectFlatline:
    """Tests cho detect_flatline."""

    def test_detects_known_flatline(
        self, detector: AnomalyDetector, flatline_signal: np.ndarray
    ) -> None:
        """Phải phát hiện vùng flatline đã biết."""
        flatline = detector.detect_flatline(flatline_signal, window=50, threshold=0.01)
        # Vùng 200:260 phải được detect (sau một chút vì rolling)
        flatline_in_region = flatline[220:260]
        assert np.sum(flatline_in_region) > 0, "Không phát hiện flatline trong vùng 220:260"

    def test_no_flatline_in_varying_signal(
        self, detector: AnomalyDetector, normal_signal: np.ndarray
    ) -> None:
        """Tín hiệu biến đổi không nên có nhiều flatline."""
        flatline = detector.detect_flatline(normal_signal, window=50, threshold=0.001)
        # Std của normal_signal rất lớn so với threshold 0.001
        # Nên không có hoặc rất ít flatline
        flatline_rate = flatline.mean()
        assert flatline_rate < 0.5  # Nhỏ hơn 50%

    def test_output_shape(
        self, detector: AnomalyDetector, flatline_signal: np.ndarray
    ) -> None:
        """Output shape phải bằng input."""
        flatline = detector.detect_flatline(flatline_signal)
        assert flatline.shape == flatline_signal.shape


# ==============================================================================
# Tests: detect_iqr_outliers
# ==============================================================================

class TestDetectIqrOutliers:
    """Tests cho detect_iqr_outliers."""

    def test_detects_extreme_values(self, detector: AnomalyDetector) -> None:
        """Phải phát hiện giá trị cực trị."""
        data = np.array([1.0] * 100 + [1000.0, -1000.0])
        outliers = detector.detect_iqr_outliers(data)
        assert outliers[-1], "Không phát hiện outlier tại -1"
        assert outliers[-2], "Không phát hiện outlier tại -2"

    def test_uniform_distribution(self, detector: AnomalyDetector) -> None:
        """Phân phối đều không nên có outlier với IQR."""
        # Dữ liệu đều: IQR sẽ bao phủ mọi điểm
        data = np.linspace(0, 100, 200)
        outliers = detector.detect_iqr_outliers(data, multiplier=1.5)
        # Có thể có một ít outlier ở biên, nhưng phải ít
        assert outliers.mean() < 0.05

    def test_output_is_boolean(self, detector: AnomalyDetector, normal_signal: np.ndarray) -> None:
        """Output phải boolean."""
        outliers = detector.detect_iqr_outliers(normal_signal)
        assert outliers.dtype == bool


# ==============================================================================
# Tests: detect_std_deviation
# ==============================================================================

class TestDetectStdDeviation:
    """Tests cho detect_std_deviation."""

    def test_detects_deviating_points(self, detector: AnomalyDetector) -> None:
        """Phải phát hiện điểm lệch khỏi rolling mean."""
        np.random.seed(42)
        signal = np.random.normal(0, 1, 500)
        signal[250] = 100.0  # Điểm lệch rất lớn
        anomalies = detector.detect_std_deviation(signal, window=50, n_std=3)
        assert anomalies[250], "Không phát hiện điểm lệch tại index 250"

    def test_output_shape(self, detector: AnomalyDetector, normal_signal: np.ndarray) -> None:
        """Output shape phải bằng input."""
        anomalies = detector.detect_std_deviation(normal_signal)
        assert anomalies.shape == normal_signal.shape


# ==============================================================================
# Tests: detect_signal_drift
# ==============================================================================

class TestDetectSignalDrift:
    """Tests cho detect_signal_drift."""

    def test_detects_strong_drift(
        self, detector: AnomalyDetector, drift_signal: np.ndarray
    ) -> None:
        """Phải phát hiện drift rõ ràng (step-change ở 8% cuối, window=40)."""
        drift = detector.detect_signal_drift(drift_signal, window=40)
        # Cuối tín hiệu (có drift lớn) phải được detect
        assert np.sum(drift) > 0, "Không phát hiện drift"

    def test_output_shape(
        self, detector: AnomalyDetector, drift_signal: np.ndarray
    ) -> None:
        """Output shape phải bằng input."""
        drift = detector.detect_signal_drift(drift_signal)
        assert drift.shape == drift_signal.shape


# ==============================================================================
# Tests: Isolation Forest và LOF
# ==============================================================================

class TestIsolationForestAndLOF:
    """Tests cho Isolation Forest và LOF."""

    def test_isolation_forest_output_shape(
        self, detector: AnomalyDetector, scada_df: pd.DataFrame
    ) -> None:
        """Isolation Forest output shape phải bằng số hàng."""
        anomalies = detector.detect_with_isolation_forest(scada_df, contamination=0.05)
        assert anomalies.shape == (len(scada_df),)

    def test_isolation_forest_detects_outlier(
        self, detector: AnomalyDetector, scada_df: pd.DataFrame
    ) -> None:
        """Isolation Forest phải phát hiện ít nhất một outlier."""
        anomalies = detector.detect_with_isolation_forest(scada_df, contamination=0.05)
        assert np.sum(anomalies) > 0

    def test_isolation_forest_contamination_rate(
        self, detector: AnomalyDetector, scada_df: pd.DataFrame
    ) -> None:
        """Tỉ lệ anomaly phải gần contamination."""
        contamination = 0.1
        anomalies = detector.detect_with_isolation_forest(scada_df, contamination=contamination)
        actual_rate = anomalies.mean()
        # Cho phép sai lệch ±5%
        assert abs(actual_rate - contamination) < 0.05

    def test_lof_output_shape(
        self, detector: AnomalyDetector, scada_df: pd.DataFrame
    ) -> None:
        """LOF output shape phải bằng số hàng."""
        anomalies = detector.detect_with_lof(scada_df, n_neighbors=10)
        assert anomalies.shape == (len(scada_df),)

    def test_lof_is_boolean(
        self, detector: AnomalyDetector, scada_df: pd.DataFrame
    ) -> None:
        """LOF output phải boolean."""
        anomalies = detector.detect_with_lof(scada_df, n_neighbors=10)
        assert anomalies.dtype == bool

    def test_empty_dataframe(self, detector: AnomalyDetector) -> None:
        """DataFrame không có cột số phải trả về zeros."""
        df_empty = pd.DataFrame({"text": ["a", "b", "c"]})
        result = detector.detect_with_isolation_forest(df_empty)
        assert not np.any(result)


# ==============================================================================
# Tests: classify_anomaly_type
# ==============================================================================

class TestClassifyAnomalyType:
    """Tests cho classify_anomaly_type."""

    def test_output_shape(
        self, detector: AnomalyDetector, spiked_signal: np.ndarray
    ) -> None:
        """Output shape phải bằng input."""
        labels = detector.classify_anomaly_type(spiked_signal)
        assert labels.shape == spiked_signal.shape

    def test_contains_valid_labels(
        self, detector: AnomalyDetector, spiked_signal: np.ndarray
    ) -> None:
        """Labels phải thuộc tập hợp hợp lệ."""
        valid_labels = {"normal", "spike", "missing", "drift", "flatline", "std_deviation"}
        labels = detector.classify_anomaly_type(spiked_signal)
        unique_labels = set(labels)
        assert unique_labels.issubset(valid_labels), f"Labels không hợp lệ: {unique_labels - valid_labels}"

    def test_spike_labeled_correctly(
        self, detector: AnomalyDetector, spiked_signal: np.ndarray
    ) -> None:
        """Vị trí spike phải được gán nhãn 'spike'."""
        labels = detector.classify_anomaly_type(spiked_signal)
        assert labels[100] == "spike", f"Spike tại 100 được nhãn '{labels[100]}'"

    def test_normal_signal_mostly_normal(
        self, detector: AnomalyDetector, normal_signal: np.ndarray
    ) -> None:
        """Tín hiệu bình thường phải có nhiều 'normal' hơn."""
        labels = detector.classify_anomaly_type(normal_signal)
        normal_rate = np.sum(labels == "normal") / len(labels)
        assert normal_rate > 0.95  # > 95% normal


# ==============================================================================
# Tests: generate_anomaly_report
# ==============================================================================

class TestGenerateAnomalyReport:
    """Tests cho generate_anomaly_report."""

    def test_returns_dataframe(
        self, detector: AnomalyDetector, scada_df: pd.DataFrame
    ) -> None:
        """Phải trả về DataFrame."""
        report = detector.generate_anomaly_report(scada_df)
        assert isinstance(report, pd.DataFrame)

    def test_report_has_required_columns(
        self, detector: AnomalyDetector, scada_df: pd.DataFrame
    ) -> None:
        """Report phải có đủ cột bắt buộc."""
        report = detector.generate_anomaly_report(scada_df)
        if not report.empty:
            required_cols = {"timestamp", "column", "type", "severity", "value"}
            assert required_cols.issubset(set(report.columns))

    def test_severity_values_valid(
        self, detector: AnomalyDetector, scada_df: pd.DataFrame
    ) -> None:
        """Severity phải thuộc {'high', 'medium', 'low'}."""
        report = detector.generate_anomaly_report(scada_df)
        if not report.empty:
            valid_severities = {"high", "medium", "low"}
            assert set(report["severity"].unique()).issubset(valid_severities)

    def test_detect_known_anomaly(self, detector: AnomalyDetector) -> None:
        """Report phải detect spike đã biết."""
        np.random.seed(42)
        idx = pd.date_range("2023-01-01", periods=200, freq="10min")
        df = pd.DataFrame({
            "signal": np.random.normal(100, 5, 200),
        }, index=idx)
        df.iloc[100, 0] = 10000.0  # Spike rõ ràng

        report = detector.generate_anomaly_report(df, signal_cols=["signal"])
        assert not report.empty, "Không phát hiện anomaly nào"
        assert "spike" in report["type"].values

    def test_empty_signal_cols(
        self, detector: AnomalyDetector, scada_df: pd.DataFrame
    ) -> None:
        """Cột không tồn tại phải được bỏ qua."""
        report = detector.generate_anomaly_report(
            scada_df, signal_cols=["nonexistent_col"]
        )
        assert isinstance(report, pd.DataFrame)
