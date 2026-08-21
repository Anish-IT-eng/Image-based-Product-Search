#!/usr/bin/env python3
"""
scripts/export_catalog_csv.py
------------------------------
Export the built catalog.json to a CSV for easy inspection in Excel/pandas.

Run from project root:
    python scripts/export_catalog_csv.py

Output: backend/data/catalog_export.csv
"""

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from config import settings

CATALOG_PATH = settings.data_dir / "catalog.json"
OUTPUT_PATH  = settings.data_dir / "catalog_export.csv"

FIELDS = [
    "cid", "brand", "category", "subcategory",
    "gender", "material", "heel_height", "closure", "toe_style",
]


def main() -> int:
    if not CATALOG_PATH.exists():
        print(f"[FAIL] Catalog not found: {CATALOG_PATH}")
        print("       Run backend/build_index.py first.")
        return 1

    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(catalog)

    rel = OUTPUT_PATH.relative_to(PROJECT_ROOT)
    print(f"[OK]   Exported {len(catalog):,} products -> {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
