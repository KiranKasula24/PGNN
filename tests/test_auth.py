from fastapi.testclient import TestClient

from pignn.service.main import app


def test_every_service_route_requires_the_internal_backend_secret():
    assert TestClient(app).get("/health").status_code == 401
    assert TestClient(app, headers={"X-GEOARGUS-INTERNAL-KEY": "wrong"}).get("/health").status_code == 401
    assert TestClient(app, headers={"X-GEOARGUS-INTERNAL-KEY": "test-internal-key"}).get("/health").status_code == 200


def test_openapi_exposes_locked_response_contracts_for_internal_consumers():
    schema = TestClient(app, headers={"X-GEOARGUS-INTERNAL-KEY": "test-internal-key"}).get("/openapi.json")
    assert schema.status_code == 200
    paths = schema.json()["paths"]
    assert "$ref" in paths["/expected-state/state"]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert "$ref" in paths["/risk-synthesis"]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
