"""
build_index.py
--------------
One-time offline script to:
  1. Walk the ut-zap50k-images directory and sample images (up to MAX_IMAGES)
  2. Extract ResNet50 embeddings for all images (CPU, batched)
  3. Build a FAISS IndexFlatIP over the embeddings
  4. Save index.faiss + catalog.json to backend/data/

Run this ONCE before starting the FastAPI server:
    python backend/build_index.py

Progress will be printed to stdout. On a modern CPU this takes ~5-10 minutes
for 1,600 images.
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np

# Make sure backend/ is importable when run from project root
sys.path.insert(0, str(Path(__file__).parent))

# Force UTF-8 output on Windows to avoid UnicodeEncodeError with arrow chars
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from catalog import IMAGES_ROOT, build_catalog, save_catalog
from config import settings
from embedder import get_embedder
from indexer import FaissIndex

# ── Config (from environment / .env) ─────────────────────────────────────────
MAX_IMAGES  = settings.max_images if settings.max_images > 0 else None
BATCH_SIZE  = settings.embedding_batch_size
DATA_DIR    = settings.data_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_index")


def main():
    start_total = time.time()

    # ── 1. Build catalog ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Image-based Product Search -- Index Builder")
    print(f"  Catalog size: {MAX_IMAGES} images | Device: CPU")
    print(f"{'='*60}\n")

    print("Step 1/4  Building catalog ...")
    catalog = build_catalog(max_images=MAX_IMAGES)
    print(f"          -> {len(catalog)} products found\n")

    if len(catalog) == 0:
        print(f"ERROR: No images found under {IMAGES_ROOT}")
        print("Make sure the ut-zap50k-images directory is present.")
        sys.exit(1)

    # ── 2. Load model ─────────────────────────────────────────────────────────
    print("Step 2/4  Loading ResNet50 model ...")
    embedder = get_embedder()
    print("          -> Model ready\n")

    image_paths = [entry["image_path"] for entry in catalog]

    # ── 3. Extract embeddings ─────────────────────────────────────────────────
    print(f"Step 3/4  Extracting embeddings for {len(image_paths)} images ...")
    print(f"          Batch size: {BATCH_SIZE} | This may take a few minutes ...\n")
    t0 = time.time()
    embeddings = embedder.embed_batch(image_paths, batch_size=BATCH_SIZE, show_progress=True)
    elapsed = time.time() - t0
    rate = len(image_paths) / elapsed if elapsed > 0 else 0
    print(f"\n          -> Done in {elapsed:.1f}s  ({rate:.1f} img/s)\n")

    # Save raw embeddings (useful for re-indexing without re-embedding)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(DATA_DIR / "embeddings.npy"), embeddings)
    print(f"          -> Embeddings saved to backend/data/embeddings.npy\n")

    # ── 4. Build & save FAISS index ───────────────────────────────────────────
    print("Step 4/4  Building FAISS index ...")
    faiss_index = FaissIndex()
    faiss_index.build(embeddings, catalog)
    faiss_index.save(
        index_path=DATA_DIR / "index.faiss",
        catalog_path=DATA_DIR / "catalog.json",
    )
    print(f"          -> index.faiss  saved to backend/data/")
    print(f"          -> catalog.json saved to backend/data/\n")

    total = time.time() - start_total
    stats = faiss_index.stats()
    print(f"{'='*60}")
    print(f"  Build complete in {total:.1f}s")
    print(f"  Indexed vectors : {stats['total_vectors']}")
    print(f"  Embedding dim   : {stats['embedding_dim']}")
    print(f"  Index type      : {stats['index_type']}")
    print(f"{'='*60}")
    print(f"\n  You can now start the API server:")
    print(f"  uvicorn main:app --reload --port 8000\n")


if __name__ == "__main__":
    main()
