"""
main.py
-------
FastAPI application for Image-based Product Search.

Endpoints:
  GET  /health         — liveness check
  GET  /stats          — catalog & index statistics
  POST /search         — upload image, get top-K similar products
  GET  /images/{path}  — serve catalog images as static files

Start with:
    uvicorn main:app --reload --port 8000
"""

import base64
import io
import logging
import mimetypes
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from cache import query_cache
from config import settings
from embedder import get_embedder
from indexer import get_index

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
_ALLOWED_MIME_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
_NORM_MIME = {"image/jpg": "image/jpeg"}  # normalise non-standard MIME

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ── Lifespan: pre-load model + index on startup ───────────────────────────────
@asynccontextmanager
async def lifespan(app):
    """Load heavy resources once at startup; release on shutdown."""
    logger.info("Loading ResNet50 embedder ...")
    get_embedder()
    logger.info("Loading FAISS index ...")
    get_index()
    logger.info("Server ready — http://localhost:8000/docs")
    yield
    logger.info("Shutting down.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Image-based Product Search API",
    description="Visual similarity search for footwear using ResNet50 + FAISS",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # 'null' covers file:// origins (opening index.html directly in browser)
    allow_origins=settings.cors_origins + ["null"],
    allow_origin_regex=r"http://localhost:\d+",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Response models ───────────────────────────────────────────────────────────
class ProductResult(BaseModel):
    cid: str
    similarity: float
    category: str
    subcategory: str
    brand: str
    gender: str
    material: str
    heel_height: str
    closure: str
    toe_style: str
    image_url: str       # URL the frontend can fetch for the product image
    image_b64: Optional[str] = None  # inline base64 thumbnail (optional)


class SearchResponse(BaseModel):
    query_time_ms: float
    total_results: int
    results: List[ProductResult]


class HealthResponse(BaseModel):
    status: str
    message: str


# ── Helpers ───────────────────────────────────────────────────────────────────
def _image_to_base64(image_path: str, size: tuple = (200, 200)) -> Optional[str]:
    """Create a small base64 thumbnail so the frontend doesn't need a separate fetch."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img.thumbnail(size, Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=75)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        logger.warning(f"Thumbnail generation failed for {image_path}: {e}")
        return None


def _make_image_url(image_path: str) -> str:
    """Convert absolute disk path to an API URL the frontend can hit."""
    # We encode the path as a URL-safe base64 string to avoid path traversal issues
    encoded = base64.urlsafe_b64encode(image_path.encode()).decode()
    return f"/images/{encoded}"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return {"status": "ok", "message": "Image-based Product Search API is running."}


@app.get("/stats", tags=["System"])
async def stats():
    try:
        index = get_index()
        data: Dict[str, Any] = index.stats()
        data.update(query_cache.stats())
        return data
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/catalog/categories", tags=["Catalog"])
async def catalog_categories():
    """
    Return the distinct categories and subcategories present in the loaded catalog.
    Used by the frontend to populate filter chips dynamically.

    Response shape:
        {
          "categories":    ["Boots", "Shoes", "Sandals", "Slippers"],
          "subcategories": {"Boots": ["Ankle", "Mid-Calf", ...], ...},
          "genders":       ["Men", "Women", "Boys", "Girls", "Unisex"]
        }
    """
    try:
        index = get_index()
        catalog = index.catalog
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    cats: set = set()
    subcats: Dict[str, set] = {}
    genders: set = set()

    for entry in catalog:
        cat = (entry.get("category") or "").strip()
        sub = (entry.get("subcategory") or "").strip()
        gen = entry.get("gender") or ""

        if cat:
            cats.add(cat)
            if cat not in subcats:
                subcats[cat] = set()
            if sub:
                subcats[cat].add(sub)

        for g in gen.split(";"):
            g = g.strip()
            if g:
                genders.add(g)

    return {
        "categories":    sorted(cats),
        "subcategories": {c: sorted(s) for c, s in sorted(subcats.items())},
        "genders":       sorted(genders),
    }


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(
    file: UploadFile = File(..., description="Query image (jpg/png/webp)"),
    top_k: int = Form(default=12, ge=1, le=50, description="Number of results"),
):
    """
    Upload a product image and receive the top-K visually similar products
    from the catalog, ranked by cosine similarity.
    """
    # ── Validate file type ────────────────────────────────────────────────────
    raw_mime = (file.content_type or "").lower()
    mime = _NORM_MIME.get(raw_mime, raw_mime)   # normalise image/jpg → image/jpeg
    if mime not in _ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG, PNG, or WebP.",
        )

    # ── Read bytes (check size before processing) ─────────────────────────────
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Cannot read upload: {e}")

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large: {len(contents) / 1_048_576:.1f} MB. "
                f"Maximum allowed size is {MAX_UPLOAD_BYTES // 1_048_576} MB."
            ),
        )

    # ── Cache check ───────────────────────────────────────────────────────────
    cached = query_cache.get(contents, top_k)
    if cached is not None:
        logger.info(f"/search cache HIT — returning cached result (top_k={top_k})")
        response = JSONResponse(content=cached)
        response.headers["X-Cache-Hit"] = "true"
        return response

    # ── Parse image ───────────────────────────────────────────────────────────
    try:
        query_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, Exception) as e:
        raise HTTPException(status_code=422, detail=f"Cannot decode image: {e}")

    # ── Embed query image ─────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        embedder = get_embedder()
        query_embedding = embedder.embed_single(query_image)
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise HTTPException(status_code=500, detail="Embedding extraction failed.")

    # ── FAISS search ──────────────────────────────────────────────────────────
    try:
        index = get_index()
        raw_results = index.search(query_embedding, top_k=top_k)
    except Exception as e:
        logger.error(f"FAISS search failed: {e}")
        raise HTTPException(status_code=500, detail="Search failed.")

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # ── Build response ────────────────────────────────────────────────────────
    results = []
    for product, similarity in raw_results:
        img_path = product.get("image_path", "")
        b64_thumb = _image_to_base64(img_path)
        image_url = _make_image_url(img_path) if img_path else ""

        results.append(
            ProductResult(
                cid=product.get("cid", ""),
                similarity=round(similarity, 4),
                category=product.get("category", ""),
                subcategory=product.get("subcategory", ""),
                brand=product.get("brand", ""),
                gender=product.get("gender", ""),
                material=product.get("material", ""),
                heel_height=product.get("heel_height", ""),
                closure=product.get("closure", ""),
                toe_style=product.get("toe_style", ""),
                image_url=image_url,
                image_b64=b64_thumb,
            )
        )

    logger.info(
        f"/search completed in {elapsed_ms:.1f}ms — "
        f"{len(results)} results for uploaded image"
    )

    response_data = SearchResponse(
        query_time_ms=round(elapsed_ms, 2),
        total_results=len(results),
        results=results,
    )

    # ── Store in cache ────────────────────────────────────────────────────────
    response_dict = response_data.model_dump()
    query_cache.set(contents, top_k, response_dict)

    response = JSONResponse(content=response_dict)
    response.headers["X-Cache-Hit"] = "false"
    return response


@app.get("/images/{encoded_path}", tags=["Images"])
async def serve_image(encoded_path: str):
    """
    Serve a catalog image by its base64-encoded absolute disk path.
    This keeps file paths off the URL surface while still letting the
    frontend fetch product images directly.
    """
    try:
        image_path = base64.urlsafe_b64decode(encoded_path.encode()).decode()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image path encoding.")

    # Security: only serve files that actually exist under the images root
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")

    # Resolve to check it's under the expected images directory
    try:
        images_root = (Path(__file__).parent.parent / "ut-zap50k-images").resolve()
        path.resolve().relative_to(images_root)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied.")

    mime, _ = mimetypes.guess_type(str(path))
    return FileResponse(str(path), media_type=mime or "image/jpeg")
