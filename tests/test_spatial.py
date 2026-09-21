import numpy as np

from pignn.spatial import local_metres, nearest_grid_id


def test_nearest_grid_join_uses_registered_node_coordinates():
    rows = [
        {"grid_id": "far", "latitude": 20.02, "longitude": 80.02},
        {"grid_id": "near", "latitude": 20.0001, "longitude": 80.0001},
    ]
    assert nearest_grid_id(20.0, 80.0, rows)["grid_id"] == "near"


def test_local_projection_is_zero_at_its_declared_origin():
    point = local_metres(np.asarray([20.0]), np.asarray([80.0]), 20.0, 80.0)
    np.testing.assert_allclose(point, [[0.0, 0.0]])
