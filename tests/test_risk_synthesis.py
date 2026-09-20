import pytest

from pignn.risk_synthesis import synthesize


def test_agreeing_high_signals_produce_high_risk_without_blending_agreement():
    result = synthesize(0.8, 0.9, low_days=2, high_days=4)
    assert result.risk_score == pytest.approx(0.98)
    assert result.recommended_action == "emergency_review"
    assert result.confidence_badge == "agreeing"
    assert result.low_days == 2


def test_a_single_strong_signal_can_raise_risk_but_reports_disagreement():
    result = synthesize(0.95, 0.05)
    assert result.risk_score == pytest.approx(0.9525)
    assert result.recommended_action == "emergency_review"
    assert result.confidence_badge == "disagreeing"


def test_risk_synthesis_rejects_invalid_inputs_and_time_windows():
    with pytest.raises(ValueError):
        synthesize(-0.1, 0.2)
    with pytest.raises(ValueError):
        synthesize(0.2, 0.3, low_days=3, high_days=2)
