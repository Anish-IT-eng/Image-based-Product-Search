"""
catalog.py
----------
Walks the ut-zap50k-images directory tree to build a unified product catalog.

The UT-Zappos50K image directory structure encodes brand information:
    <IMAGES_ROOT>/<Category>/<SubCategory>/<Brand>/<filename>.jpg

Metadata priority:
  1. CID  — derived from image filename stem via _extract_cid_from_path()
  2. Brand — derived from the immediate parent directory name via _extract_brand_from_path()
  3. Category / SubCategory / Gender / Material / Closure / HeelHeight / ToeStyle
     — sourced from meta-data.csv when a CID match exists; defaults to "Unknown" / ""
       when no CSV row matches (expected for most images in this extract).

Helper functions:
  _extract_cid_from_path(path)   → path.stem
  _extract_brand_from_path(path) → path.parent.name
  _load_metadata()               → dict keyed by CID string
  build_catalog(max_images)      → List[dict]
  save_catalog(catalog, path)    → None
  load_catalog(path)             → List[dict]
"""

import csv
import json
import logging
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

# Canonical paths — sourced from centralised config (override via .env)
IMAGES_ROOT  = settings.images_dir
METADATA_CSV = settings.metadata_csv
CATALOG_JSON = settings.data_dir / "catalog.json"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _extract_cid_from_path(image_path: Path) -> str:
    """
    Extract the product CID from an image file path for CSV lookup.

    Image filenames use '.' as separator (e.g. "7965307.5291.jpg")
    but the CSV uses '-' (e.g. "7965307-5291"). Replace '.' with '-'
    so the lookup succeeds.

    Example:
        Path("ut-zap50k-images/Boots/Ankle/Nike/7965307.5291.jpg")
        → "7965307-5291"  (matches CSV CID column)
    """
    return image_path.stem.replace(".", "-")


def _extract_brand_from_path(image_path: Path) -> str:
    """
    Extract the brand name from an image file path.

    Directory structure: <Category>/<SubCategory>/<Brand>/<filename>.jpg
    The brand is the immediate parent directory of the image file.

    Example:
        Path("/root/Boots/Ankle/Clarks/101.jpg") → "Clarks"
    """
    return image_path.parent.name


def _extract_category_from_path(image_path: Path, images_root: Path) -> str:
    """
    Extract category from directory structure as a fallback.

    Structure: <images_root>/<Category>/<SubCategory>/<Brand>/<file>
    Returns the first path component after images_root.
    """
    try:
        rel = image_path.relative_to(images_root)
        return rel.parts[0] if rel.parts else "Unknown"
    except ValueError:
        return "Unknown"


def _extract_subcategory_from_path(image_path: Path, images_root: Path) -> str:
    """
    Extract subcategory from directory structure as a fallback.

    Structure: <images_root>/<Category>/<SubCategory>/<Brand>/<file>
    Returns the second path component after images_root.
    """
    try:
        rel = image_path.relative_to(images_root)
        return rel.parts[1] if len(rel.parts) > 1 else "Unknown"
    except ValueError:
        return "Unknown"


def _load_metadata() -> dict:
    """
    Parse meta-data.csv → dict keyed by CID string.

    Returns an empty dict if the CSV is not found (non-fatal — catalog entries
    will then fall back to "Unknown" for category/subcategory and "" for
    gender/material/closure/heel_height/toe_style).
    """
    meta: dict = {}
    if not METADATA_CSV.exists():
        logger.warning(
            f"Metadata CSV not found at {METADATA_CSV} — "
            "category/subcategory will default to 'Unknown'."
        )
        return meta

    with open(METADATA_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row.get("CID", "").strip()
            if cid:
                meta[cid] = {
                    "category":    row.get("Category", "").strip(),
                    "subcategory": row.get("SubCategory", "").strip(),
                    "heel_height": row.get("HeelHeight", "").strip(),
                    "insole":      row.get("Insole", "").strip(),
                    "closure":     row.get("Closure", "").strip(),
                    "gender":      row.get("Gender", "").strip(),
                    "material":    row.get("Material", "").strip(),
                    "toe_style":   row.get("ToeStyle", "").strip(),
                }

    logger.info(f"Loaded metadata for {len(meta)} products from CSV")
    return meta


# ── Public API ────────────────────────────────────────────────────────────────

def build_catalog(max_images: int = None) -> list:
    """
    Walk the image directory and build a unified product catalog.

    For each image:
      - CID        is extracted from the filename stem
      - Brand      is extracted from the immediate parent directory name
      - Category   is sourced from meta-data.csv (keyed by CID);
                   defaults to "Unknown" when no CSV row matches
      - SubCategory, Gender, Material, Closure, HeelHeight, ToeStyle
                   likewise sourced from CSV; default to "" when not found

    Args:
        max_images: Cap the catalog at this many images (0 / None = no limit).

    Returns:
        List of dicts: {cid, image_path, brand, category, subcategory,
                        gender, material, closure, heel_height, toe_style}
    """
    if not IMAGES_ROOT.exists():
        raise FileNotFoundError(
            f"Images directory not found: {IMAGES_ROOT}\n"
            "Extract ut-zap50k-images.zip to the project root first."
        )

    csv_meta = _load_metadata()
    catalog:  list = []

    logger.info(f"Walking image tree at {IMAGES_ROOT} ...")

    for img_path in sorted(IMAGES_ROOT.rglob("*.jpg")):
        cid   = _extract_cid_from_path(img_path)
        brand = _extract_brand_from_path(img_path)

        # CSV lookup — provides category/subcategory + extended attributes
        # CID in CSV uses '-'; image stems use '.' → already converted by _extract_cid_from_path
        extra = csv_meta.get(cid, {})

        # Fallback: derive category/subcategory from directory structure
        # Path structure: <images_root>/<Category>/<SubCategory>/<Brand>/<file.jpg>
        # This ensures 0% "Unknown" even for CIDs not present in the CSV.
        path_category    = _extract_category_from_path(img_path, IMAGES_ROOT)
        path_subcategory = _extract_subcategory_from_path(img_path, IMAGES_ROOT)

        entry = {
            "cid":          cid,
            "image_path":   str(img_path),          # absolute disk path
            "brand":        brand,
            "category":     extra.get("category",    path_category),
            "subcategory":  extra.get("subcategory", path_subcategory),
            "heel_height":  extra.get("heel_height", ""),
            "closure":      extra.get("closure",     ""),
            "gender":       extra.get("gender",      ""),
            "material":     extra.get("material",    ""),
            "toe_style":    extra.get("toe_style",   ""),
        }
        catalog.append(entry)

        if max_images and len(catalog) >= max_images:
            break

    logger.info(f"Catalog built: {len(catalog)} products")
    return catalog


def save_catalog(catalog: list, path: Path = None) -> None:
    """Persist catalog to JSON for use by the API at runtime."""
    out_path = path or CATALOG_JSON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)
    logger.info(f"Catalog saved → {out_path}  ({len(catalog)} products)")


def load_catalog(path: Path = None) -> list:
    """Load catalog from the persisted JSON file."""
    in_path = path or CATALOG_JSON
    if not in_path.exists():
        raise FileNotFoundError(
            f"Catalog JSON not found at {in_path}.\n"
            "Run `python backend/build_index.py` first."
        )
    with open(in_path, encoding="utf-8") as f:
        catalog = json.load(f)
    logger.info(f"Loaded catalog: {len(catalog)} products from {in_path}")
    return catalog
