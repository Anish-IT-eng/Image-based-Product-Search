"""
test_indexer.py
---------------
Unit tests for FaissIndex — build, search, save/load round-trip, and stats.

Uses only numpy + faiss (no PyTorch, no disk index files needed except for
the save/load tests which use pytest's tmp_path fixture).
"""

import json
import numpy as np
import pytest
from pathlib import Path

from indexer import FaissIndex

EMBEDDING_DIM = 2048
N = 10


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def embeddings() -> np.ndarray:
    """N L2-normalized random float32 vectors."""
    rng = np.random.default_rng(seed=1)
    emb = rng.standard_normal((N, EMBEDDING_DIM)).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / norms


@pytest.fixture
def catalog() -> list:
    return [
        {
            "cid": f"cid-{i:03d}",
            "image_path": "",
            "brand": f"Brand{i}",
            "category": "Boots" if i % 2 == 0 else "Shoes",
            "subcategory": "Ankle",
            "gender": "Women",
            "material": "Leather",
            "heel_height": "Low",
            "closure": "Lace-Up",
            "toe_style": "Round",
        }
        for i in range(N)
    ]


@pytest.fixture
def built_index(embeddings, catalog) -> FaissIndex:
    idx = FaissIndex()
    idx.build(embeddings, catalog)
    return idx


# ── Build ──────────────────────────────────────────────────────────────────────

class TestBuild:
    def test_ntotal_equals_n(self, built_index):
        assert built_index.index.ntotal == N

    def test_catalog_length_equals_n(self, built_index):
        assert len(built_index.catalog) == N

    def test_ready_flag_set_after_build(self, built_index):
        assert built_index._ready is True

    def test_fresh_index_is_not_ready(self):
        assert FaissIndex()._ready is False

    def test_build_asserts_mismatched_lengths(self):
        idx = FaissIndex()
        emb = np.ones((5, EMBEDDING_DIM), dtype=np.float32)
        with pytest.raises(AssertionError):
            idx.build(emb, [{"cid": "x"}] * 3)   # catalog length != 5

    def test_build_drops_zero_norm_rows(self, catalog):
        """Rows with zero L2-norm (failed images) must be excluded from the index."""
        rng = np.random.default_rng(seed=2)
        emb = rng.standard_normal((N, EMBEDDING_DIM)).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / norms
        emb[2] = 0.0   # simulate failed image
        emb[7] = 0.0   # simulate failed image

        idx = FaissIndex()
        idx.build(emb, catalog)
        assert idx.index.ntotal == N - 2
        assert len(idx.catalog) == N - 2

    def test_build_requires_float32(self, catalog):
        emb = np.ones((N, EMBEDDING_DIM), dtype=np.float64)  # wrong dtype
        idx = FaissIndex()
        with pytest.raises(AssertionError):
            idx.build(emb, catalog)


# ── Search ─────────────────────────────────────────────────────────────────────

class TestSearch:
    def test_returns_list_of_tuples(self, built_index, embeddings):
        results = built_index.search(embeddings[0], top_k=3)
        assert isinstance(results, list)
        assert len(results) == 3
        for product, score in results:
            assert isinstance(product, dict)
            assert isinstance(score, float)

    def test_scores_are_descending(self, built_index, embeddings):
        results = built_index.search(embeddings[0], top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_scores_in_valid_range(self, built_index, embeddings):
        results = built_index.search(embeddings[0], top_k=N)
        for _, score in results:
            assert 0.0 <= score <= 1.0

    def test_self_similarity_is_top_result(self, built_index, embeddings):
        """Querying with catalog vector[0] must return cid-000 as top hit."""
        results = built_index.search(embeddings[0], top_k=1)
        top_product, top_score = results[0]
        assert top_product["cid"] == "cid-000"
        assert abs(top_score - 1.0) < 1e-4   # cosine sim of vector with itself ≈ 1

    def test_top_k_clamped_to_index_size(self, built_index, embeddings):
        results = built_index.search(embeddings[0], top_k=99999)
        assert len(results) <= N

    def test_top_k_1_returns_one_result(self, built_index, embeddings):
        results = built_index.search(embeddings[0], top_k=1)
        assert len(results) == 1

    def test_1d_query_accepted(self, built_index, embeddings):
        """Query vector must work both as (D,) and (1, D) shape."""
        q_1d = embeddings[0]           # shape (D,)
        assert q_1d.ndim == 1
        results = built_index.search(q_1d, top_k=3)
        assert len(results) == 3

    def test_search_on_unloaded_index_raises(self):
        idx = FaissIndex()
        with pytest.raises(RuntimeError, match="not loaded"):
            idx.search(np.zeros(EMBEDDING_DIM, dtype=np.float32))

    def test_result_products_have_cid_key(self, built_index, embeddings):
        results = built_index.search(embeddings[0], top_k=3)
        for product, _ in results:
            assert "cid" in product


# ── Persistence ────────────────────────────────────────────────────────────────

class TestPersistence:
    def test_save_creates_index_file(self, built_index, tmp_path):
        idx_path = tmp_path / "index.faiss"
        cat_path = tmp_path / "catalog.json"
        built_index.save(index_path=idx_path, catalog_path=cat_path)
        assert idx_path.exists()

    def test_save_creates_catalog_file(self, built_index, tmp_path):
        idx_path = tmp_path / "index.faiss"
        cat_path = tmp_path / "catalog.json"
        built_index.save(index_path=idx_path, catalog_path=cat_path)
        assert cat_path.exists()

    def test_catalog_json_is_valid(self, built_index, tmp_path):
        cat_path = tmp_path / "catalog.json"
        built_index.save(
            index_path=tmp_path / "index.faiss",
            catalog_path=cat_path,
        )
        with open(cat_path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == N

    def test_save_load_roundtrip_ntotal(self, built_index, tmp_path):
        idx_path = tmp_path / "index.faiss"
        cat_path = tmp_path / "catalog.json"
        built_index.save(index_path=idx_path, catalog_path=cat_path)

        loaded = FaissIndex()
        loaded.load(index_path=idx_path, catalog_path=cat_path)
        assert loaded.index.ntotal == N

    def test_save_load_roundtrip_search_results(self, built_index, embeddings, tmp_path):
        idx_path = tmp_path / "index.faiss"
        cat_path = tmp_path / "catalog.json"
        built_index.save(index_path=idx_path, catalog_path=cat_path)

        loaded = FaissIndex()
        loaded.load(index_path=idx_path, catalog_path=cat_path)

        q = embeddings[0]
        orig_cids = [p["cid"] for p, _ in built_index.search(q, top_k=3)]
        loaded_cids = [p["cid"] for p, _ in loaded.search(q, top_k=3)]
        assert orig_cids == loaded_cids

    def test_load_missing_index_raises(self, tmp_path):
        idx = FaissIndex()
        with pytest.raises(FileNotFoundError, match="FAISS index not found"):
            idx.load(
                index_path=tmp_path / "missing.faiss",
                catalog_path=tmp_path / "catalog.json",
            )

    def test_load_missing_catalog_raises(self, built_index, tmp_path):
        idx_path = tmp_path / "index.faiss"
        built_index.save(
            index_path=idx_path,
            catalog_path=tmp_path / "catalog.json",
        )
        idx = FaissIndex()
        with pytest.raises(FileNotFoundError, match="Catalog JSON not found"):
            idx.load(
                index_path=idx_path,
                catalog_path=tmp_path / "nonexistent_catalog.json",
            )


# ── Stats ──────────────────────────────────────────────────────────────────────

class TestStats:
    def test_stats_before_build_returns_not_loaded(self):
        idx = FaissIndex()
        assert idx.stats()["status"] == "not_loaded"

    def test_stats_after_build_is_ready(self, built_index):
        assert built_index.stats()["status"] == "ready"

    def test_stats_total_vectors(self, built_index):
        assert built_index.stats()["total_vectors"] == N

    def test_stats_catalog_size(self, built_index):
        assert built_index.stats()["catalog_size"] == N

    def test_stats_embedding_dim(self, built_index):
        assert built_index.stats()["embedding_dim"] == EMBEDDING_DIM

    def test_stats_has_index_type(self, built_index):
        assert "index_type" in built_index.stats()
