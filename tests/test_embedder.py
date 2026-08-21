"""
test_embedder.py
----------------
Unit tests for ResNet50Embedder and _l2_normalize.

The PyTorch model is replaced with a MagicMock so tests run instantly
with no model weights download and no GPU requirement.
"""

import io
import torch
import numpy as np
import pytest
from PIL import Image
from unittest.mock import MagicMock, patch

from embedder import EMBEDDING_DIM, _l2_normalize


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_pil(w: int = 32, h: int = 32, color=(128, 64, 32)) -> Image.Image:
    return Image.new("RGB", (w, h), color=color)


def _make_jpeg_path(tmp_path, name: str = "img.jpg") -> str:
    p = tmp_path / name
    _make_pil().save(str(p), format="JPEG")
    return str(p)


def _make_mocked_embedder():
    """
    Build a ResNet50Embedder whose internal nn.Sequential is replaced by a
    MagicMock.  The mock returns a (1, EMBEDDING_DIM, 1, 1) tensor by default.
    Avoids all weight downloading and GPU usage.
    """
    from embedder import ResNet50Embedder

    fake_out = torch.ones(1, EMBEDDING_DIM, 1, 1) * 0.5

    mock_backbone = MagicMock()
    with (
        patch("embedder.models.resnet50", return_value=mock_backbone),
        patch("embedder.models.ResNet50_Weights"),
    ):
        emb = object.__new__(ResNet50Embedder)
        emb.device = torch.device("cpu")
        emb.model = MagicMock(return_value=fake_out)
        emb.model.eval.return_value = emb.model
        emb.model.to.return_value = emb.model

    return emb


def _make_batch_embedder(batch_size: int):
    """Embedder whose mock scales its output to match the batch size."""
    from embedder import ResNet50Embedder

    def _fake_forward(tensor: torch.Tensor):
        B = tensor.shape[0]
        return torch.ones(B, EMBEDDING_DIM, 1, 1) * 0.3

    mock_backbone = MagicMock()
    with (
        patch("embedder.models.resnet50", return_value=mock_backbone),
        patch("embedder.models.ResNet50_Weights"),
    ):
        emb = object.__new__(ResNet50Embedder)
        emb.device = torch.device("cpu")
        emb.model = MagicMock(side_effect=_fake_forward)
        emb.model.eval.return_value = emb.model
        emb.model.to.return_value = emb.model

    return emb


# ── _l2_normalize ──────────────────────────────────────────────────────────────

class TestL2Normalize:
    def test_unit_vector_unchanged(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = _l2_normalize(v)
        np.testing.assert_allclose(result, v, atol=1e-6)

    def test_normalizes_to_unit_norm(self):
        v = np.array([3.0, 4.0], dtype=np.float32)   # norm = 5
        result = _l2_normalize(v)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6

    def test_known_values(self):
        v = np.array([3.0, 4.0], dtype=np.float32)
        result = _l2_normalize(v)
        np.testing.assert_allclose(result, [0.6, 0.8], atol=1e-6)

    def test_zero_vector_returned_as_is(self):
        v = np.zeros(8, dtype=np.float32)
        result = _l2_normalize(v)
        np.testing.assert_array_equal(result, v)

    def test_near_zero_vector_returned_as_is(self):
        v = np.full(4, 1e-8, dtype=np.float32)
        result = _l2_normalize(v)
        # norm < 1e-6, so returned unchanged
        np.testing.assert_array_equal(result, v)


# ── preprocess ─────────────────────────────────────────────────────────────────

class TestPreprocess:
    def test_output_shape(self):
        from embedder import _TRANSFORM
        img = _make_pil(64, 64)
        tensor = _TRANSFORM(img).unsqueeze(0)
        assert tensor.shape == (1, 3, 224, 224)

    def test_output_dtype(self):
        from embedder import _TRANSFORM
        tensor = _TRANSFORM(_make_pil()).unsqueeze(0)
        assert tensor.dtype == torch.float32

    def test_preprocess_accepts_pil(self):
        emb = _make_mocked_embedder()
        tensor = emb.preprocess(_make_pil())
        assert tensor.shape == (1, 3, 224, 224)

    def test_preprocess_accepts_file_path(self, tmp_path):
        emb = _make_mocked_embedder()
        p = tmp_path / "test.jpg"
        _make_pil().save(str(p), format="JPEG")
        tensor = emb.preprocess(str(p))
        assert tensor.shape == (1, 3, 224, 224)

    def test_preprocess_converts_rgba_to_rgb(self):
        emb = _make_mocked_embedder()
        rgba = Image.new("RGBA", (32, 32), (255, 0, 0, 128))
        tensor = emb.preprocess(rgba)
        assert tensor.shape == (1, 3, 224, 224)


# ── embed_single ───────────────────────────────────────────────────────────────

class TestEmbedSingle:
    def test_output_shape(self):
        emb = _make_mocked_embedder()
        vec = emb.embed_single(_make_pil())
        assert vec.shape == (EMBEDDING_DIM,)

    def test_output_dtype_float32(self):
        emb = _make_mocked_embedder()
        vec = emb.embed_single(_make_pil())
        assert vec.dtype == np.float32

    def test_output_is_l2_normalized(self):
        emb = _make_mocked_embedder()
        vec = emb.embed_single(_make_pil())
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5

    def test_accepts_file_path(self, tmp_path):
        emb = _make_mocked_embedder()
        p = _make_jpeg_path(tmp_path)
        vec = emb.embed_single(p)
        assert vec.shape == (EMBEDDING_DIM,)

    def test_returns_numpy_array(self):
        emb = _make_mocked_embedder()
        vec = emb.embed_single(_make_pil())
        assert isinstance(vec, np.ndarray)


# ── embed_batch ────────────────────────────────────────────────────────────────

class TestEmbedBatch:
    def test_output_shape_3_images(self, tmp_path):
        emb = _make_batch_embedder(batch_size=2)
        paths = [_make_jpeg_path(tmp_path, f"img{i}.jpg") for i in range(3)]
        result = emb.embed_batch(paths, batch_size=2, show_progress=False)
        assert result.shape == (3, EMBEDDING_DIM)

    def test_output_dtype_float32(self, tmp_path):
        emb = _make_batch_embedder(batch_size=2)
        paths = [_make_jpeg_path(tmp_path, f"img{i}.jpg") for i in range(2)]
        result = emb.embed_batch(paths, batch_size=2, show_progress=False)
        assert result.dtype == np.float32

    def test_rows_are_l2_normalized(self, tmp_path):
        emb = _make_batch_embedder(batch_size=4)
        paths = [_make_jpeg_path(tmp_path, f"img{i}.jpg") for i in range(4)]
        result = emb.embed_batch(paths, batch_size=4, show_progress=False)
        for row in result:
            norm = np.linalg.norm(row)
            assert abs(norm - 1.0) < 1e-5, f"Row norm was {norm}"

    def test_single_image_batch(self, tmp_path):
        emb = _make_batch_embedder(batch_size=1)
        paths = [_make_jpeg_path(tmp_path, "single.jpg")]
        result = emb.embed_batch(paths, batch_size=1, show_progress=False)
        assert result.shape == (1, EMBEDDING_DIM)

    def test_failed_image_produces_zero_row(self, tmp_path):
        """A corrupt/missing image path should yield a zero vector row."""
        emb = _make_batch_embedder(batch_size=2)
        good_path = _make_jpeg_path(tmp_path, "good.jpg")
        bad_path = str(tmp_path / "nonexistent.jpg")   # doesn't exist

        result = emb.embed_batch([good_path, bad_path], batch_size=2, show_progress=False)
        assert result.shape == (2, EMBEDDING_DIM)
        # The bad row should be zero (failed open → zero tensor → norm < 1e-6 → left as zero)
        assert np.linalg.norm(result[1]) < 1e-6
