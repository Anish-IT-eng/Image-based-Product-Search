"""
test_health.py
--------------
Tests for the system-level endpoints:
  GET /health   — liveness check
  GET /stats    — catalog & index statistics
"""


class TestHealthEndpoint:
    def test_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_returns_ok_status(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"

    def test_returns_message_field(self, client):
        body = client.get("/health").json()
        assert "message" in body
        assert isinstance(body["message"], str)
        assert len(body["message"]) > 0


class TestStatsEndpoint:
    def test_returns_200(self, client):
        response = client.get("/stats")
        assert response.status_code == 200

    def test_status_is_ready(self, client):
        body = client.get("/stats").json()
        assert body["status"] == "ready"

    def test_catalog_size_matches_mock(self, client):
        # N_CATALOG = 5 from conftest.py
        body = client.get("/stats").json()
        assert body["catalog_size"] == 5

    def test_total_vectors_matches_catalog(self, client):
        body = client.get("/stats").json()
        assert body["total_vectors"] == body["catalog_size"]

    def test_embedding_dim_is_2048(self, client):
        body = client.get("/stats").json()
        assert body["embedding_dim"] == 2048

    def test_index_type_present(self, client):
        body = client.get("/stats").json()
        assert "index_type" in body
