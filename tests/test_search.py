"""
test_search.py
--------------
Tests for POST /search — the core product similarity search endpoint.

All tests use the mocked embedder + in-memory FAISS index from conftest.py,
so no model weights or on-disk index files are required.
"""

import pytest


class TestSearchValidInputs:
    def test_jpeg_returns_200(self, client, sample_jpeg_bytes):
        response = client.post(
            "/search",
            files={"file": ("shoe.jpg", sample_jpeg_bytes, "image/jpeg")},
            data={"top_k": "5"},
        )
        assert response.status_code == 200

    def test_png_returns_200(self, client, sample_png_bytes):
        response = client.post(
            "/search",
            files={"file": ("shoe.png", sample_png_bytes, "image/png")},
            data={"top_k": "3"},
        )
        assert response.status_code == 200

    def test_response_contains_results_list(self, client, sample_jpeg_bytes):
        body = client.post(
            "/search",
            files={"file": ("shoe.jpg", sample_jpeg_bytes, "image/jpeg")},
            data={"top_k": "5"},
        ).json()
        assert "results" in body
        assert isinstance(body["results"], list)
        assert len(body["results"]) > 0

    def test_response_contains_query_time_ms(self, client, sample_jpeg_bytes):
        body = client.post(
            "/search",
            files={"file": ("shoe.jpg", sample_jpeg_bytes, "image/jpeg")},
            data={"top_k": "5"},
        ).json()
        assert "query_time_ms" in body
        assert body["query_time_ms"] >= 0

    def test_response_contains_total_results(self, client, sample_jpeg_bytes):
        body = client.post(
            "/search",
            files={"file": ("shoe.jpg", sample_jpeg_bytes, "image/jpeg")},
            data={"top_k": "5"},
        ).json()
        assert "total_results" in body
        assert body["total_results"] == len(body["results"])

    def test_default_top_k_used_when_omitted(self, client, sample_jpeg_bytes):
        """Without top_k, default=12 is used; catalog has only 5 items."""
        body = client.post(
            "/search",
            files={"file": ("shoe.jpg", sample_jpeg_bytes, "image/jpeg")},
        ).json()
        assert len(body["results"]) <= 5   # capped by catalog size


class TestSearchTopK:
    def test_top_k_1_returns_one_result(self, client, sample_jpeg_bytes):
        body = client.post(
            "/search",
            files={"file": ("shoe.jpg", sample_jpeg_bytes, "image/jpeg")},
            data={"top_k": "1"},
        ).json()
        assert len(body["results"]) == 1

    def test_top_k_3_returns_at_most_3(self, client, sample_jpeg_bytes):
        body = client.post(
            "/search",
            files={"file": ("shoe.jpg", sample_jpeg_bytes, "image/jpeg")},
            data={"top_k": "3"},
        ).json()
        assert len(body["results"]) <= 3

    def test_top_k_exceeds_api_limit_rejected(self, client, sample_jpeg_bytes):
        """top_k > 50 violates the FastAPI Form validator (ge=1, le=50) → 422."""
        response = client.post(
            "/search",
            files={"file": ("shoe.jpg", sample_jpeg_bytes, "image/jpeg")},
            data={"top_k": "9999"},
        )
        assert response.status_code == 422

    def test_top_k_at_api_limit_accepted(self, client, sample_jpeg_bytes):
        """top_k=50 is the maximum allowed value — clamped to catalog size (5)."""
        body = client.post(
            "/search",
            files={"file": ("shoe.jpg", sample_jpeg_bytes, "image/jpeg")},
            data={"top_k": "50"},
        ).json()
        assert len(body["results"]) <= 5   # N_CATALOG

    def test_top_k_zero_rejected(self, client, sample_jpeg_bytes):
        response = client.post(
            "/search",
            files={"file": ("shoe.jpg", sample_jpeg_bytes, "image/jpeg")},
            data={"top_k": "0"},
        )
        assert response.status_code == 422


class TestSearchResultSchema:
    """Every ProductResult must contain all required fields."""

    REQUIRED_FIELDS = {
        "cid", "similarity", "category", "subcategory",
        "brand", "gender", "material", "heel_height",
        "closure", "toe_style", "image_url",
    }

    def test_result_has_all_required_fields(self, client, sample_jpeg_bytes):
        body = client.post(
            "/search",
            files={"file": ("shoe.jpg", sample_jpeg_bytes, "image/jpeg")},
            data={"top_k": "1"},
        ).json()
        result = body["results"][0]
        assert self.REQUIRED_FIELDS.issubset(result.keys())

    def test_similarity_is_float_in_range(self, client, sample_jpeg_bytes):
        body = client.post(
            "/search",
            files={"file": ("shoe.jpg", sample_jpeg_bytes, "image/jpeg")},
            data={"top_k": "5"},
        ).json()
        for r in body["results"]:
            assert isinstance(r["similarity"], float)
            assert 0.0 <= r["similarity"] <= 1.0

    def test_cid_is_non_empty_string(self, client, sample_jpeg_bytes):
        body = client.post(
            "/search",
            files={"file": ("shoe.jpg", sample_jpeg_bytes, "image/jpeg")},
            data={"top_k": "3"},
        ).json()
        for r in body["results"]:
            assert isinstance(r["cid"], str)
            assert len(r["cid"]) > 0


class TestSearchRejectedInputs:
    def test_text_plain_rejected_422(self, client):
        response = client.post(
            "/search",
            files={"file": ("doc.txt", b"hello world", "text/plain")},
            data={"top_k": "5"},
        )
        assert response.status_code == 422

    def test_corrupt_jpeg_bytes_rejected_422(self, client):
        response = client.post(
            "/search",
            files={"file": ("bad.jpg", b"definitely-not-an-image", "image/jpeg")},
            data={"top_k": "5"},
        )
        assert response.status_code == 422

    def test_empty_file_rejected_422(self, client):
        response = client.post(
            "/search",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
            data={"top_k": "5"},
        )
        assert response.status_code == 422

    def test_missing_file_field_rejected_422(self, client):
        response = client.post("/search", data={"top_k": "5"})
        assert response.status_code == 422


class TestImageServingEndpoint:
    def test_invalid_base64_returns_400(self, client):
        response = client.get("/images/!!!not_valid_base64!!!")
        assert response.status_code == 400

    def test_nonexistent_path_returns_404(self, client):
        import base64
        encoded = base64.urlsafe_b64encode(b"/nonexistent/path/img.jpg").decode()
        response = client.get(f"/images/{encoded}")
        assert response.status_code == 404
