"""
conftest.py
-----------
Shared pytest fixtures for the VisualFind test suite.

All fixtures are fully isolated:
  - No live FastAPI server needed (uses TestClient)
  - No FAISS index on disk (builds an in-memory index)
  - No ResNet50 weights downloaded (embedder is mocked)
  - No disk images needed (PIL generates them in-memory)
"""

import io
import numpy as np
import pytest
from PIL import Image
from unittest.mock import MagicMock, patch

EMBEDDING_DIM = 2048
N_CATALOG = 5           # size of the in-memory mock catalog


# ── In-memory test images ──────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_jpeg_bytes() -> bytes:
    """Minimal valid 10×10 red JPEG — no disk I/O."""
    img = Image.new("RGB", (10, 10), color=(220, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def sample_png_bytes() -> bytes:
    """Minimal valid 10×10 green PNG — no disk I/O."""
    img = Image.new("RGB", (10, 10), color=(50, 200, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Mock embedder ──────────────────────────────────────────────────────────────

@pytest.fixture
def mock_embedder():
    """
    Fake ResNet50Embedder that returns a fixed L2-normalized 2048-dim vector.
    Avoids any PyTorch model loading during tests.
    """
    rng = np.random.default_rng(seed=42)
    vec = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    vec /= np.linalg.norm(vec)

    embedder = MagicMock()
    embedder.embed_single.return_value = vec
    return embedder


# ── Mock FAISS index ───────────────────────────────────────────────────────────

@pytest.fixture
def mock_faiss_index():
    """
    In-memory FaissIndex built from N_CATALOG dummy L2-normalized embeddings.
    Safe to use without any on-disk index file.
    """
    from indexer import FaissIndex

    rng = np.random.default_rng(seed=0)
    embeddings = rng.standard_normal((N_CATALOG, EMBEDDING_DIM)).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    catalog = [
        {
            "cid": f"test-{i:04d}",
            "image_path": "",           # empty — thumbnail generation will return None
            "brand": f"Brand{i}",
            "category": "Boots" if i % 2 == 0 else "Shoes",
            "subcategory": "Ankle" if i % 2 == 0 else "Oxford",
            "gender": "Women",
            "material": "Leather",
            "heel_height": "Low",
            "closure": "Lace-Up",
            "toe_style": "Round",
        }
        for i in range(N_CATALOG)
    ]

    idx = FaissIndex()
    idx.build(embeddings, catalog)
    return idx


# ── FastAPI TestClient ─────────────────────────────────────────────────────────

@pytest.fixture
def client(mock_embedder, mock_faiss_index):
    """
    FastAPI TestClient with the embedder and FAISS index both mocked out.
    The lifespan startup handlers call the patched versions, so no model
    weights are loaded and no index file needs to exist on disk.
    """
    from fastapi.testclient import TestClient
    import main as main_module

    with (
        patch("main.get_embedder", return_value=mock_embedder),
        patch("main.get_index", return_value=mock_faiss_index),
    ):
        with TestClient(main_module.app) as c:
            yield c
