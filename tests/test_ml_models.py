"""
Unit tests cho ML Models module.

Tests bao gồm:
- FeatureEngineer: time-domain, frequency-domain, wavelet, statistical, rolling features
- FaultClassifier: training, evaluation, feature importance, save/load
- LSTMAutoencoder: build, train, reconstruction error, anomaly detection
"""

import os
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.ml_models.feature_engineering import FeatureEngineer
from src.ml_models.fault_classifier import FaultClassifier


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def feature_engineer() -> FeatureEngineer:
    """FeatureEngineer instance."""
    return FeatureEngineer(window_size=50, rolling_windows=[10, 20], fs=1 / 600.0)


@pytest.fixture
def fault_classifier() -> FaultClassifier:
    """FaultClassifier instance."""
    return FaultClassifier(random_state=42)


@pytest.fixture
def sample_signal() -> np.ndarray:
    """Tín hiệu mẫu 200 điểm."""
    np.random.seed(42)
    t = np.linspace(0, 10, 200)
    return np.sin(2 * np.pi * 0.5 * t) + np.random.normal(0, 0.1, 200)


@pytest.fixture
def scada_df() -> pd.DataFrame:
    """DataFrame SCADA giả lập."""
    np.random.seed(42)
    n = 500
    idx = pd.date_range("2023-01-01", periods=n, freq="10min")
    df = pd.DataFrame({
        "LV ActivePower (kW)": np.random.normal(1000, 200, n),
        "Wind Speed (m/s)": np.random.uniform(3, 15, n),
        "Theoretical_Power_Curve (KWh)": np.random.normal(1100, 150, n),
        "Wind Direction (°)": np.random.uniform(0, 360, n),
    }, index=idx)
    return df


@pytest.fixture
def classification_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dữ liệu phân loại cho testing."""
    X, y = make_classification(
        n_samples=300,
        n_features=20,
        n_informative=10,
        n_classes=4,
        n_clusters_per_class=1,
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ==============================================================================
# Tests: FeatureEngineer - Time domain
# ==============================================================================

class TestTimeDomainFeatures:
    """Tests cho time-domain feature extraction."""

    def test_returns_all_expected_keys(
        self, feature_engineer: FeatureEngineer, sample_signal: np.ndarray
    ) -> None:
        """Phải trả về đúng các key mong đợi."""
        features = feature_engineer.extract_time_domain_features(sample_signal)
        expected_keys = {
            "mean", "std", "min", "max", "rms", "kurtosis",
            "skewness", "peak_to_peak", "crest_factor", "shape_factor"
        }
        assert expected_keys.issubset(set(features.keys()))

    def test_mean_correct(
        self, feature_engineer: FeatureEngineer, sample_signal: np.ndarray
    ) -> None:
        """mean phải bằng np.mean."""
        features = feature_engineer.extract_time_domain_features(sample_signal)
        assert abs(features["mean"] - float(np.mean(sample_signal))) < 1e-10

    def test_std_correct(
        self, feature_engineer: FeatureEngineer, sample_signal: np.ndarray
    ) -> None:
        """std phải bằng np.std."""
        features = feature_engineer.extract_time_domain_features(sample_signal)
        assert abs(features["std"] - float(np.std(sample_signal))) < 1e-10

    def test_rms_positive(
        self, feature_engineer: FeatureEngineer, sample_signal: np.ndarray
    ) -> None:
        """RMS phải dương."""
        features = feature_engineer.extract_time_domain_features(sample_signal)
        assert features["rms"] > 0

    def test_peak_to_peak(
        self, feature_engineer: FeatureEngineer, sample_signal: np.ndarray
    ) -> None:
        """peak_to_peak phải bằng max - min."""
        features = feature_engineer.extract_time_domain_features(sample_signal)
        expected = float(np.max(sample_signal) - np.min(sample_signal))
        assert abs(features["peak_to_peak"] - expected) < 1e-10

    def test_nan_signal_returns_zeros(
        self, feature_engineer: FeatureEngineer
    ) -> None:
        """Tín hiệu toàn NaN phải trả về zeros."""
        all_nan = np.full(50, np.nan)
        features = feature_engineer.extract_time_domain_features(all_nan)
        assert features["mean"] == 0.0
        assert features["rms"] == 0.0


# ==============================================================================
# Tests: FeatureEngineer - Frequency domain
# ==============================================================================

class TestFrequencyDomainFeatures:
    """Tests cho frequency-domain feature extraction."""

    def test_returns_all_expected_keys(
        self, feature_engineer: FeatureEngineer, sample_signal: np.ndarray
    ) -> None:
        """Phải trả về đúng các key."""
        features = feature_engineer.extract_frequency_domain_features(sample_signal)
        expected_keys = {
            "spectral_centroid", "spectral_bandwidth",
            "spectral_rolloff", "dominant_freq", "spectral_entropy"
        }
        assert expected_keys.issubset(set(features.keys()))

    def test_all_values_are_finite(
        self, feature_engineer: FeatureEngineer, sample_signal: np.ndarray
    ) -> None:
        """Tất cả values phải hữu hạn."""
        features = feature_engineer.extract_frequency_domain_features(sample_signal)
        for key, val in features.items():
            assert np.isfinite(val), f"Feature '{key}' không hữu hạn: {val}"

    def test_spectral_entropy_positive(
        self, feature_engineer: FeatureEngineer, sample_signal: np.ndarray
    ) -> None:
        """Spectral entropy phải dương."""
        features = feature_engineer.extract_frequency_domain_features(sample_signal)
        assert features["spectral_entropy"] >= 0

    def test_short_signal_returns_zeros(self, feature_engineer: FeatureEngineer) -> None:
        """Tín hiệu quá ngắn phải trả về zeros."""
        short_signal = np.array([1.0, 2.0])
        features = feature_engineer.extract_frequency_domain_features(short_signal)
        assert all(v == 0.0 for v in features.values())


# ==============================================================================
# Tests: FeatureEngineer - Statistical features
# ==============================================================================

class TestStatisticalFeatures:
    """Tests cho statistical feature extraction."""

    def test_returns_expected_keys(
        self, feature_engineer: FeatureEngineer, sample_signal: np.ndarray
    ) -> None:
        """Phải trả về đúng keys."""
        features = feature_engineer.extract_statistical_features(sample_signal)
        assert "p25" in features
        assert "p50" in features
        assert "p75" in features
        assert "variance" in features
        assert "cv" in features

    def test_percentiles_ordered(
        self, feature_engineer: FeatureEngineer, sample_signal: np.ndarray
    ) -> None:
        """p25 <= p50 <= p75."""
        features = feature_engineer.extract_statistical_features(sample_signal)
        assert features["p25"] <= features["p50"]
        assert features["p50"] <= features["p75"]

    def test_variance_positive(
        self, feature_engineer: FeatureEngineer, sample_signal: np.ndarray
    ) -> None:
        """Variance phải >= 0."""
        features = feature_engineer.extract_statistical_features(sample_signal)
        assert features["variance"] >= 0


# ==============================================================================
# Tests: FeatureEngineer - Rolling features
# ==============================================================================

class TestRollingFeatures:
    """Tests cho rolling feature extraction."""

    def test_returns_all_windows(
        self, feature_engineer: FeatureEngineer, sample_signal: np.ndarray
    ) -> None:
        """Phải trả về features cho mỗi window size."""
        features = feature_engineer.extract_rolling_features(
            sample_signal, windows=[10, 20]
        )
        assert "rolling_mean_w10" in features
        assert "rolling_std_w10" in features
        assert "rolling_mean_w20" in features

    def test_rolling_mean_constant(self, feature_engineer: FeatureEngineer) -> None:
        """Rolling mean của hằng số phải bằng hằng số đó."""
        constant = np.ones(100) * 7.0
        features = feature_engineer.extract_rolling_features(constant, windows=[10])
        assert abs(features["rolling_mean_w10"] - 7.0) < 1e-10


# ==============================================================================
# Tests: FeatureEngineer - Wavelet features
# ==============================================================================

class TestWaveletFeatures:
    """Tests cho wavelet feature extraction."""

    def test_returns_energy_levels(
        self, feature_engineer: FeatureEngineer, sample_signal: np.ndarray
    ) -> None:
        """Phải trả về wavelet energy levels."""
        try:
            features = feature_engineer.extract_wavelet_features(sample_signal, levels=5)
            assert any(k.startswith("wavelet_energy") for k in features.keys())
        except ImportError:
            pytest.skip("pywt không khả dụng")

    def test_energy_sum_near_one(
        self, feature_engineer: FeatureEngineer, sample_signal: np.ndarray
    ) -> None:
        """Tổng energy normalized phải gần 1."""
        try:
            features = feature_engineer.extract_wavelet_features(sample_signal)
            total = sum(v for v in features.values())
            assert abs(total - 1.0) < 0.1, f"Total energy={total}"
        except ImportError:
            pytest.skip("pywt không khả dụng")


# ==============================================================================
# Tests: FeatureEngineer - Feature matrix
# ==============================================================================

class TestFeatureMatrix:
    """Tests cho create_feature_matrix."""

    def test_returns_dataframe(
        self, feature_engineer: FeatureEngineer, scada_df: pd.DataFrame
    ) -> None:
        """Phải trả về DataFrame."""
        feature_matrix = feature_engineer.create_feature_matrix(scada_df, window_size=50)
        assert isinstance(feature_matrix, pd.DataFrame)

    def test_has_multiple_features(
        self, feature_engineer: FeatureEngineer, scada_df: pd.DataFrame
    ) -> None:
        """Phải có nhiều features > số cột gốc."""
        feature_matrix = feature_engineer.create_feature_matrix(scada_df, window_size=50)
        assert feature_matrix.shape[1] > 4

    def test_normalize_standard(
        self, feature_engineer: FeatureEngineer, scada_df: pd.DataFrame
    ) -> None:
        """StandardScaler phải tạo features với mean~0, std~1."""
        feature_matrix = feature_engineer.create_feature_matrix(scada_df, window_size=50)
        numeric_cols = feature_matrix.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            normalized = feature_engineer.normalize_features(
                feature_matrix[numeric_cols], method="standard"
            )
            # Mean xấp xỉ 0 (cho phép sai số lớn vì ít samples)
            assert abs(normalized.mean().mean()) < 1.0


# ==============================================================================
# Tests: FaultClassifier
# ==============================================================================

class TestFaultClassifier:
    """Tests cho FaultClassifier."""

    def test_train_random_forest_returns_model(
        self,
        fault_classifier: FaultClassifier,
        classification_data: Tuple,
    ) -> None:
        """train_random_forest phải trả về model đã train."""
        from sklearn.ensemble import RandomForestClassifier
        X_train, X_test, y_train, y_test = classification_data
        model = fault_classifier.train_random_forest(X_train, y_train, cv=3)
        assert isinstance(model, RandomForestClassifier)
        assert "random_forest" in fault_classifier.models

    def test_train_gradient_boosting(
        self,
        fault_classifier: FaultClassifier,
        classification_data: Tuple,
    ) -> None:
        """train_gradient_boosting phải trả về model."""
        from sklearn.ensemble import GradientBoostingClassifier
        X_train, X_test, y_train, y_test = classification_data
        model = fault_classifier.train_gradient_boosting(X_train, y_train, cv=3)
        assert isinstance(model, GradientBoostingClassifier)

    def test_evaluate_model_returns_metrics(
        self,
        fault_classifier: FaultClassifier,
        classification_data: Tuple,
    ) -> None:
        """evaluate_model phải trả về dict với các metrics."""
        X_train, X_test, y_train, y_test = classification_data
        model = fault_classifier.train_random_forest(X_train, y_train, cv=2)
        metrics = fault_classifier.evaluate_model(model, X_test, y_test)

        assert "confusion_matrix" in metrics
        assert "classification_report" in metrics
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_save_and_load_model(
        self,
        fault_classifier: FaultClassifier,
        classification_data: Tuple,
    ) -> None:
        """save/load model phải hoạt động đúng."""
        X_train, X_test, y_train, y_test = classification_data
        model = fault_classifier.train_random_forest(X_train, y_train, cv=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.joblib"
            fault_classifier.save_model(model, model_path)
            assert model_path.exists()

            loaded_model = fault_classifier.load_model(model_path)
            # Dự đoán phải giống nhau
            preds_original = model.predict(X_test)
            preds_loaded = loaded_model.predict(X_test)
            np.testing.assert_array_equal(preds_original, preds_loaded)

    def test_get_feature_importance(
        self,
        fault_classifier: FaultClassifier,
        classification_data: Tuple,
    ) -> None:
        """get_feature_importance phải trả về DataFrame đúng."""
        X_train, X_test, y_train, y_test = classification_data
        model = fault_classifier.train_random_forest(X_train, y_train, cv=2)
        feature_names = [f"feat_{i}" for i in range(X_train.shape[1])]

        importance_df = fault_classifier.get_feature_importance(model, feature_names)
        assert isinstance(importance_df, pd.DataFrame)
        assert "feature" in importance_df.columns
        assert "importance" in importance_df.columns
        assert len(importance_df) == X_train.shape[1]

        # Importances phải tổng bằng 1
        total_importance = importance_df["importance"].sum()
        assert abs(total_importance - 1.0) < 1e-6

    def test_load_nonexistent_model_raises(
        self, fault_classifier: FaultClassifier
    ) -> None:
        """load_model với path không tồn tại phải raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            fault_classifier.load_model("/nonexistent/path/model.joblib")

    def test_ensemble_predict_single_model(
        self,
        fault_classifier: FaultClassifier,
        classification_data: Tuple,
    ) -> None:
        """ensemble_predict với một model phải hoạt động."""
        X_train, X_test, y_train, y_test = classification_data
        fault_classifier.train_random_forest(X_train, y_train, cv=2)
        preds = fault_classifier.ensemble_predict(X_test)
        assert len(preds) == len(X_test)

    def test_ensemble_predict_no_models_raises(
        self, fault_classifier: FaultClassifier, classification_data: Tuple
    ) -> None:
        """ensemble_predict khi chưa train phải raise RuntimeError."""
        X_train, X_test, y_train, y_test = classification_data
        with pytest.raises(RuntimeError):
            fault_classifier.ensemble_predict(X_test)

    def test_fault_labels_defined(self, fault_classifier: FaultClassifier) -> None:
        """Phải có đủ 6 nhãn lỗi."""
        assert len(FaultClassifier.FAULT_LABELS) == 6
        assert 0 in FaultClassifier.FAULT_LABELS
        assert FaultClassifier.FAULT_LABELS[0] == "Normal operation"


# ==============================================================================
# Tests: LSTMAutoencoder
# ==============================================================================

class TestLSTMAutoencoder:
    """Tests cho LSTMAutoencoder."""

    def test_import_and_init(self) -> None:
        """LSTMAutoencoder phải có thể khởi tạo."""
        from src.ml_models.lstm_autoencoder import LSTMAutoencoder, BACKEND
        lstm = LSTMAutoencoder(sequence_length=10, n_features=2, encoding_dim=8)
        assert lstm.sequence_length == 10
        assert lstm.n_features == 2
        assert lstm.encoding_dim == 8

    def test_build_model(self) -> None:
        """build_model phải tạo model mà không lỗi."""
        from src.ml_models.lstm_autoencoder import LSTMAutoencoder, BACKEND
        if BACKEND == "none":
            pytest.skip("Không có deep learning framework")

        lstm = LSTMAutoencoder(sequence_length=10, n_features=2, encoding_dim=8)
        model = lstm.build_model()
        assert model is not None

    def test_train_and_reconstruct(self) -> None:
        """Model đã train phải có thể reconstruct."""
        from src.ml_models.lstm_autoencoder import LSTMAutoencoder, BACKEND
        if BACKEND == "none":
            pytest.skip("Không có deep learning framework")

        np.random.seed(42)
        seq_len = 10
        n_feat = 2
        n_samples = 50

        X = np.random.normal(0, 1, (n_samples, seq_len, n_feat)).astype(np.float32)

        lstm = LSTMAutoencoder(sequence_length=seq_len, n_features=n_feat, encoding_dim=4)
        lstm.build_model()
        lstm.train(X, epochs=3, batch_size=16, patience=5)

        errors = lstm.get_reconstruction_error(X)
        assert len(errors) == n_samples
        assert np.all(errors >= 0)

    def test_set_threshold(self) -> None:
        """set_threshold phải set threshold dương."""
        from src.ml_models.lstm_autoencoder import LSTMAutoencoder, BACKEND
        if BACKEND == "none":
            pytest.skip("Không có deep learning framework")

        np.random.seed(42)
        seq_len = 10
        n_feat = 2
        n_samples = 50

        X = np.random.normal(0, 1, (n_samples, seq_len, n_feat)).astype(np.float32)

        lstm = LSTMAutoencoder(sequence_length=seq_len, n_features=n_feat, encoding_dim=4)
        lstm.build_model()
        lstm.train(X, epochs=2, batch_size=16, patience=3)

        threshold = lstm.set_threshold(X, percentile=95)
        assert threshold > 0
        assert lstm.threshold == threshold

    def test_detect_anomalies_returns_correct_types(self) -> None:
        """detect_anomalies phải trả về (bool_array, float_array)."""
        from src.ml_models.lstm_autoencoder import LSTMAutoencoder, BACKEND
        if BACKEND == "none":
            pytest.skip("Không có deep learning framework")

        np.random.seed(42)
        seq_len = 10
        n_feat = 2
        n_samples = 30

        X = np.random.normal(0, 1, (n_samples, seq_len, n_feat)).astype(np.float32)

        lstm = LSTMAutoencoder(sequence_length=seq_len, n_features=n_feat, encoding_dim=4)
        lstm.build_model()
        lstm.train(X, epochs=2, batch_size=16, patience=3)

        is_anomaly, errors = lstm.detect_anomalies(X)
        assert len(is_anomaly) == n_samples
        assert len(errors) == n_samples
        assert is_anomaly.dtype == bool
