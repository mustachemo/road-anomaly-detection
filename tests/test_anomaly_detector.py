"""Tests for the anomaly detector module."""

import numpy as np
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.models.anomaly_detector import AnomalyDetector
from src.features.processor import FeatureProcessor


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    # Generate normal data from a multivariate normal distribution
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    mean = np.zeros(n_features)
    cov = np.eye(n_features)
    normal_data = np.random.multivariate_normal(mean, cov, n_samples)

    # Generate some anomalies
    n_anomalies = 10
    anomaly_data = np.random.multivariate_normal(
        mean + 3,  # Shifted mean
        cov * 2,  # Larger variance
        n_anomalies,
    )

    return np.vstack([normal_data, anomaly_data])


def test_anomaly_detector_initialization():
    """Test anomaly detector initialization with different parameters."""
    # Test valid parameters
    detector = AnomalyDetector(kernel="rbf", nu=0.1, gamma="scale")
    assert detector.kernel == "rbf"
    assert detector.nu == 0.1
    assert detector.gamma == "scale"
    assert detector.model is None

    # Test invalid kernel
    with pytest.raises(ValueError):
        AnomalyDetector(kernel="invalid")

    # Test invalid nu
    with pytest.raises(ValueError):
        AnomalyDetector(nu=2.0)

    # Test invalid gamma
    with pytest.raises(ValueError):
        AnomalyDetector(gamma="invalid")


def test_anomaly_detector_fit_predict(sample_data):
    """Test model fitting and prediction."""
    detector = AnomalyDetector()

    # Fit the model
    detector.fit(sample_data)
    assert detector.model is not None

    # Make predictions
    predictions = detector.predict(sample_data)
    assert predictions.shape == (len(sample_data),)
    assert set(np.unique(predictions)) == {-1, 1}  # -1 for anomalies, 1 for normal


def test_anomaly_detector_decision_function(sample_data):
    """Test decision function computation."""
    detector = AnomalyDetector()
    detector.fit(sample_data)

    # Compute decision function
    decision_values = detector.decision_function(sample_data)
    assert decision_values.shape == (len(sample_data),)
    assert not np.all(np.isnan(decision_values))


def test_anomaly_detector_save_load(sample_data):
    """Test model saving and loading."""
    detector = AnomalyDetector()
    detector.fit(sample_data)

    with TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.joblib"

        # Save model
        detector.save(model_path)
        assert model_path.exists()

        # Load model
        loaded_detector = AnomalyDetector.load(model_path)
        assert loaded_detector.model is not None

        # Compare predictions
        original_predictions = detector.predict(sample_data)
        loaded_predictions = loaded_detector.predict(sample_data)
        np.testing.assert_array_equal(original_predictions, loaded_predictions)


def test_feature_processor_integration(sample_data):
    """Test integration with feature processor."""
    # Initialize components
    processor = FeatureProcessor(n_components=5)
    detector = AnomalyDetector()

    # Process data
    processed_data = processor.fit_transform(sample_data)
    assert processed_data.shape[1] == 5  # Reduced to 5 components

    # Train detector on processed data
    detector.fit(processed_data)
    predictions = detector.predict(processed_data)
    assert predictions.shape == (len(sample_data),)
    assert set(np.unique(predictions)) == {-1, 1}


def test_feature_importance(sample_data):
    """Test feature importance computation."""
    processor = FeatureProcessor(n_components=5)
    processor.fit(sample_data)

    # Get feature importance
    importance = processor.get_feature_importance()
    assert importance is not None
    assert importance.shape == (sample_data.shape[1],)
    assert np.all(importance >= 0)  # Importance scores should be non-negative


def test_explained_variance(sample_data):
    """Test explained variance ratio computation."""
    processor = FeatureProcessor(n_components=5)
    processor.fit(sample_data)

    # Get explained variance ratio
    variance_ratio = processor.get_explained_variance_ratio()
    assert variance_ratio is not None
    assert variance_ratio.shape == (5,)
    assert np.all(variance_ratio >= 0)
    assert np.all(variance_ratio <= 1)
    assert np.isclose(np.sum(variance_ratio), 1.0, atol=1e-10)
