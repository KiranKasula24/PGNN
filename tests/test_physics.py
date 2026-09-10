import numpy as np
from pignn.physics.knothe import progression
from pignn.physics.probability_integral import PIMParameters, PanelGeometry, basin_subsidence_mm


def test_knothe_reference_curve():
    values = progression(np.array([0.0, 10.0, 20.0]), rate_per_day=np.log(2) / 10)
    np.testing.assert_allclose(values, [0.0, 0.5, 0.75], rtol=1e-12)


def test_pim_symmetric_reference_case():
    panel = PanelGeometry(0, 0, 200, 300, 200, extraction_thickness_m=3)
    params = PIMParameters(0.6, 2.0)
    centre = basin_subsidence_mm(0, 0, panel, params)
    np.testing.assert_allclose(basin_subsidence_mm(50, 0, panel, params), basin_subsidence_mm(-50, 0, panel, params))
    assert 0 < centre < 1800  # cannot exceed q * extraction thickness in mm
    assert basin_subsidence_mm(2000, 2000, panel, params) < centre * 1e-20
