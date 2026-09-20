import pytest

from pignn.physics.geology_priors import explain_simulated_priors, simulated_geology_context


def test_simulated_geology_context_is_explicitly_not_real_earth_engine_data():
    result = explain_simulated_priors(20.0, 80.0)
    assert result["context"]["source"] == "simulated_prototype"
    assert result["longwall_priors"]["uncertainty_band_mm"] == 100.0
    with pytest.raises(ValueError):
        simulated_geology_context(91.0, 80.0)
