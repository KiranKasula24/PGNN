from pignn.features.insar_features import InSARSample
from pignn.features.registry import DEFAULT_REGISTRY


def test_renamed_feature_contracts_are_registered_and_validated():
    assert "fuzzy_risk_index" in DEFAULT_REGISTRY.names
    assert "insar_los_displacement_mm" in DEFAULT_REGISTRY.names
    assert InSARSample(2.5, 0.8).validated() == (2.5, True)
    assert "crack_anomaly_score" in DEFAULT_REGISTRY.names
