import pytest

from pignn.expected_state.bord_and_pillar import CPHSRParameters, Pillar, cascading_redistribution, cphsr_score, safety_factor, structural_risk_from_safety_factor


def test_safety_factor_matches_closed_form_reference():
    state = safety_factor(10.0, 5.0, 100.0, 10.0, 0.025)
    assert state.stress_mpa == pytest.approx(5.625)
    assert state.strength_mpa == pytest.approx(8.002)
    assert state.safety_factor == pytest.approx(8.002 / 5.625)
    assert state.failed is False


def test_cphsr_is_explicitly_provisional_and_monotonic_in_brittleness():
    low = cphsr_score(CPHSRParameters(2.0, 150.0, 2.5, 0.2, 2500.0))
    high = cphsr_score(CPHSRParameters(2.0, 150.0, 2.5, 0.8, 2500.0))
    assert low.coefficients_verified is False
    assert high.score_0_to_1 > low.score_0_to_1


def test_safe_safety_factor_has_no_pillar_risk_and_prototype_margin_is_not_emergency():
    prototype = safety_factor(15.0, 4.8, 150.0, 10.0, 0.024525)
    assert prototype.safety_factor == pytest.approx(1.248, abs=0.01)
    assert structural_risk_from_safety_factor(prototype.safety_factor) < 0.51
    assert structural_risk_from_safety_factor(1.5) == 0.0


def test_failed_pillar_redistributes_its_load_to_adjacent_pillars():
    pillars = [
        Pillar("P1", 5.0, 10.0, ("P2",)),
        Pillar("P2", 20.0, 2.0, ("P1", "P3")),
        Pillar("P3", 20.0, 2.0, ("P2",)),
    ]
    states = {state.pillar_id: state for state in cascading_redistribution(pillars, 100.0, 10.0, 0.025)}
    assert states["P1"].failed is True
    assert states["P1"].stress_mpa == 0.0
    # P2 also fails under the transferred load, which passes the load on to P3.
    assert states["P2"].failed is True
    assert states["P3"].stress_mpa > safety_factor(20.0, 2.0, 100.0, 10.0, 0.025).stress_mpa
