import pytest


@pytest.fixture(autouse=True)
def internal_service_key(monkeypatch):
    monkeypatch.setenv("PIGNN_INTERNAL_API_KEY", "test-internal-key")
