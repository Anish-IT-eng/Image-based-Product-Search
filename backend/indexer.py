"""
indexer.py
----------
FAISS index management for the visual search engine.

Uses IndexFlatIP (inner product) — since all embeddings are L2-normalized,
inner product equals cosine similarity. No approximation trade-offs for this
catalog size (1.6k–50k). If scaled to millions, switch to IndexIVFFlat.
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from config import settings

logger = logging.getLogger(__name__)

DATA_DIR     = settings.data_dir
INDEX_PATH   = DATA_DIR / "index.faiss"
CATALOG_JSON = DATA_DIR / "catalog.json"


class FaissIndex:
    """
    Wraps a FAISS IndexFlatIP for cosine similarity search over
    L2-normalized embedding vectors.
    """

    def __init__(self):
        self.index: faiss.Index = None
        self.catalog: list = []  # parallel list — catalog[i] ↔ index vector i
        self._ready = False

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, embeddings: np.ndarray, catalog: list) -> None:
        """
        Build the FAISS index from a (N, D) float32 embedding matrix.

        Args:
            embeddings: (N, D) array of L2-normalized float32 vectors.
            catalog:    List of N product dicts (parallel to embeddings).
        """
        assert embeddings.dtype == np.float32, "Embeddings must be float32"
        assert len(embeddings) == len(catalog), "Embeddings and catalog must have same length"

        # Filter out zero-rows (failed images during batch embedding)
        valid_mask = np.linalg.norm(embeddings, axis=1) > 1e-6
        n_invalid = int((~valid_mask).sum())
        if n_invalid:
            logger.warning(f"Dropping {n_invalid} zero embeddings (failed images).")

        embeddings = embeddings[valid_mask]
        self.catalog = [catalog[i] for i in range(len(catalog)) if valid_mask[i]]

        n, d = embeddings.shape
        logger.info(f"Building FAISS IndexFlatIP with {n} vectors of dim {d} ...")
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)
        self._ready = True
        logger.info(f"FAISS index built: {self.index.ntotal} vectors indexed.")

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self, index_path: Path = None, catalog_path: Path = None) -> None:
        """Persist FAISS index + catalog JSON to disk."""
        self._assert_ready()
        idx_path = index_path or INDEX_PATH
        cat_path = catalog_path or CATALOG_JSON
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        cat_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(idx_path))
        with open(cat_path, "w", encoding="utf-8") as f:
            json.dump(self.catalog, f, indent=2)

        logger.info(f"Index saved → {idx_path}")
        logger.info(f"Catalog saved → {cat_path}")

    def load(self, index_path: Path = None, catalog_path: Path = None) -> None:
        """Load FAISS index + catalog JSON from disk."""
        idx_path = index_path or INDEX_PATH
        cat_path = catalog_path or CATALOG_JSON

        if not idx_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {idx_path}. "
                "Run build_index.py first."
            )
        if not cat_path.exists():
            raise FileNotFoundError(
                f"Catalog JSON not found at {cat_path}. "
                "Run build_index.py first."
            )

        self.index = faiss.read_index(str(idx_path))
        with open(cat_path, encoding="utf-8") as f:
            self.catalog = json.load(f)

        self._ready = True
        logger.info(
            f"FAISS index loaded: {self.index.ntotal} vectors, "
            f"catalog: {len(self.catalog)} products"
        )

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self, query_embedding: np.ndarray, top_k: int = 12
    ) -> List[Tuple[dict, float]]:
        """
        Find the top-K most similar products to a query embedding.

        Args:
            query_embedding: (2048,) L2-normalized float32 vector.
            top_k:           Number of results to return.

        Returns:
            List of (product_dict, similarity_score) tuples, sorted descending.
        """
        self._assert_ready()

        q = query_embedding.astype(np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q, k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            product = self.catalog[idx]
            similarity = float(np.clip(score, 0.0, 1.0))
            results.append((product, similarity))

        return results

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        if not self._ready:
            return {"status": "not_loaded"}
        return {
            "status": "ready",
            "total_vectors": self.index.ntotal,
            "catalog_size": len(self.catalog),
            "embedding_dim": self.index.d,
            "index_type": type(self.index).__name__,
        }

    def _assert_ready(self):
        if not self._ready:
            raise RuntimeError("Index not loaded. Call load() or build() first.")


# ── Singleton factory ─────────────────────────────────────────────────────────
_index_instance: FaissIndex = None


def get_index() -> FaissIndex:
    """Return the singleton FaissIndex, loading from disk on first call."""
    global _index_instance
    if _index_instance is None:
        _index_instance = FaissIndex()
        _index_instance.load()
    return _index_instance
