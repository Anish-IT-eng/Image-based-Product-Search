"""
test_catalog.py
---------------
Unit tests for catalog.py — metadata parsing, path extraction helpers,
and build_catalog / save_catalog / load_catalog functions.

Uses tmp_path and monkeypatch to avoid any dependency on the real dataset.
"""

import csv
import json
import pytest
from pathlib import Path
from unittest.mock import patch

from catalog import (
    _extract_cid_from_path,
    _extract_brand_from_path,
    build_catalog,
    save_catalog,
    load_catalog,
)


# ── Path extraction helpers ────────────────────────────────────────────────────

class TestExtractCidFromPath:
    def test_standard_zappos_filename(self):
        p = Path("ut-zap50k-images/Boots/Ankle/Nike/101178-68022.jpg")
        assert _extract_cid_from_path(p) == "101178-68022"

    def test_real_zappos_dot_format_converted_to_hyphen(self):
        """Real Zappos filenames use dots (7965307.5291.jpg) but CSV uses
        hyphens (7965307-5291). _extract_cid_from_path must convert."""
        p = Path("ut-zap50k-images/Boots/Ankle/Nike/7965307.5291.jpg")
        assert _extract_cid_from_path(p) == "7965307-5291"

    def test_strips_extension(self):
        p = Path("/any/dir/my-product-abc.jpg")
        assert _extract_cid_from_path(p) == "my-product-abc"

    def test_numeric_only_cid(self):
        p = Path("/root/99999.jpg")
        assert _extract_cid_from_path(p) == "99999"

    def test_deep_nested_path(self):
        p = Path("/a/b/c/d/e/f/shoe-XYZ.jpg")
        assert _extract_cid_from_path(p) == "shoe-XYZ"


class TestExtractBrandFromPath:
    def test_brand_is_parent_directory(self):
        p = Path("/root/Boots/Ankle/Clarks/101.jpg")
        assert _extract_brand_from_path(p) == "Clarks"

    def test_various_brands(self):
        cases = [
            (Path("/r/Shoes/Oxford/Adidas/1.jpg"), "Adidas"),
            (Path("/r/Sandals/Flat/Gucci/2.jpg"), "Gucci"),
            (Path("/r/Slippers/None/Nike/3.jpg"), "Nike"),
        ]
        for path, expected in cases:
            assert _extract_brand_from_path(path) == expected


# ── build_catalog ──────────────────────────────────────────────────────────────

class TestBuildCatalog:
    def _make_image_tree(self, root: Path, n: int = 4,
                          category="Boots", subcategory="Ankle", brand="Nike"):
        brand_dir = root / category / subcategory / brand
        brand_dir.mkdir(parents=True)
        for i in range(n):
            (brand_dir / f"img{i:04d}.jpg").write_bytes(b"fake-image-data")
        return brand_dir

    def test_raises_when_images_root_missing(self, tmp_path):
        nonexistent = tmp_path / "no-such-dir"
        with patch("catalog.IMAGES_ROOT", nonexistent):
            with pytest.raises(FileNotFoundError, match="Images directory not found"):
                build_catalog()

    def test_returns_list(self, tmp_path):
        self._make_image_tree(tmp_path, n=3)
        with (
            patch("catalog.IMAGES_ROOT", tmp_path),
            patch("catalog.METADATA_CSV", tmp_path / "nonexistent.csv"),
        ):
            result = build_catalog()
        assert isinstance(result, list)

    def test_correct_count(self, tmp_path):
        self._make_image_tree(tmp_path, n=4)
        with (
            patch("catalog.IMAGES_ROOT", tmp_path),
            patch("catalog.METADATA_CSV", tmp_path / "nonexistent.csv"),
        ):
            result = build_catalog()
        assert len(result) == 4

    def test_max_images_cap_respected(self, tmp_path):
        self._make_image_tree(tmp_path, n=10)
        with (
            patch("catalog.IMAGES_ROOT", tmp_path),
            patch("catalog.METADATA_CSV", tmp_path / "nonexistent.csv"),
        ):
            result = build_catalog(max_images=3)
        assert len(result) == 3

    def test_entry_has_all_required_keys(self, tmp_path):
        self._make_image_tree(tmp_path, n=1)
        with (
            patch("catalog.IMAGES_ROOT", tmp_path),
            patch("catalog.METADATA_CSV", tmp_path / "nonexistent.csv"),
        ):
            result = build_catalog()

        entry = result[0]
        required = {
            "cid", "image_path", "brand", "category",
            "subcategory", "gender", "material",
            "heel_height", "closure", "toe_style",
        }
        assert required.issubset(entry.keys())

    def test_brand_extracted_from_directory_name(self, tmp_path):
        self._make_image_tree(tmp_path, n=1, brand="Gucci")
        with (
            patch("catalog.IMAGES_ROOT", tmp_path),
            patch("catalog.METADATA_CSV", tmp_path / "nonexistent.csv"),
        ):
            result = build_catalog()
        assert result[0]["brand"] == "Gucci"

    def test_image_path_is_absolute_string(self, tmp_path):
        self._make_image_tree(tmp_path, n=1)
        with (
            patch("catalog.IMAGES_ROOT", tmp_path),
            patch("catalog.METADATA_CSV", tmp_path / "nonexistent.csv"),
        ):
            result = build_catalog()
        path_str = result[0]["image_path"]
        assert isinstance(path_str, str)
        assert Path(path_str).is_absolute()

    def test_metadata_merged_from_csv(self, tmp_path):
        """When a CSV exists with a matching CID, metadata fields are populated."""
        self._make_image_tree(tmp_path, n=1, category="Boots",
                               subcategory="Ankle", brand="Clarks")
        # The CID is the filename stem = "img0000"
        csv_path = tmp_path / "meta.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "CID", "Category", "SubCategory", "HeelHeight",
                "Insole", "Closure", "Gender", "Material", "ToeStyle",
            ])
            writer.writeheader()
            writer.writerow({
                "CID": "img0000",
                "Category": "Boots",
                "SubCategory": "Ankle",
                "HeelHeight": "High",
                "Insole": "Memory Foam",
                "Closure": "Zipper",
                "Gender": "Women",
                "Material": "Suede",
                "ToeStyle": "Pointed",
            })

        with (
            patch("catalog.IMAGES_ROOT", tmp_path),
            patch("catalog.METADATA_CSV", csv_path),
        ):
            result = build_catalog()

        entry = result[0]
        assert entry["material"] == "Suede"
        assert entry["closure"] == "Zipper"
        assert entry["gender"] == "Women"
        assert entry["heel_height"] == "High"

    def test_missing_metadata_falls_back_to_path_derived_values(self, tmp_path):
        """When no CSV match exists, category/subcategory come from the
        directory path structure (Category/SubCategory/Brand/file.jpg),
        not from a hardcoded 'Unknown' string."""
        self._make_image_tree(tmp_path, n=1,
                              category="Boots", subcategory="Ankle")
        with (
            patch("catalog.IMAGES_ROOT", tmp_path),
            patch("catalog.METADATA_CSV", tmp_path / "nonexistent.csv"),
        ):
            result = build_catalog()
        # Category and subcategory are derived from path, not 'Unknown'
        assert result[0]["category"] == "Boots"
        assert result[0]["subcategory"] == "Ankle"

    def test_empty_directory_returns_empty_list(self, tmp_path):
        """An images root with no .jpg files should return an empty catalog."""
        empty_root = tmp_path / "empty"
        empty_root.mkdir()
        with (
            patch("catalog.IMAGES_ROOT", empty_root),
            patch("catalog.METADATA_CSV", tmp_path / "nonexistent.csv"),
        ):
            result = build_catalog()
        assert result == []


# ── save_catalog / load_catalog ────────────────────────────────────────────────

class TestSaveLoadCatalog:
    SAMPLE = [
        {"cid": "001", "brand": "Nike", "category": "Boots"},
        {"cid": "002", "brand": "Adidas", "category": "Shoes"},
    ]

    def test_save_creates_file(self, tmp_path):
        out = tmp_path / "catalog.json"
        save_catalog(self.SAMPLE, path=out)
        assert out.exists()

    def test_saved_file_is_valid_json(self, tmp_path):
        out = tmp_path / "catalog.json"
        save_catalog(self.SAMPLE, path=out)
        with open(out) as f:
            data = json.load(f)
        assert isinstance(data, list)

    def test_roundtrip_preserves_length(self, tmp_path):
        out = tmp_path / "catalog.json"
        save_catalog(self.SAMPLE, path=out)
        loaded = load_catalog(path=out)
        assert len(loaded) == len(self.SAMPLE)

    def test_roundtrip_preserves_data(self, tmp_path):
        out = tmp_path / "catalog.json"
        save_catalog(self.SAMPLE, path=out)
        loaded = load_catalog(path=out)
        assert loaded == self.SAMPLE

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Catalog JSON not found"):
            load_catalog(path=tmp_path / "ghost.json")

    def test_save_creates_parent_dirs(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c" / "catalog.json"
        save_catalog(self.SAMPLE, path=nested)
        assert nested.exists()
