"""
embedder.py
-----------
ResNet50-based image feature extractor.

Uses a pretrained ResNet50 (ImageNet weights) with the classification head
removed. The resulting 2048-dimensional global average pooling output is
L2-normalized to enable cosine similarity via inner product in FAISS.

Runs entirely on CPU (no CUDA required).
"""

import logging
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

logger = logging.getLogger(__name__)

# ── ImageNet normalization ────────────────────────────────────────────────────
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

EMBEDDING_DIM = 2048

# Path to optional fine-tuned model weights produced by train.py
_MODEL_PATH = Path(__file__).parent / "data" / "best_model.pth"


class ResNet50Embedder:
    """
    Extracts 2048-dim L2-normalized embeddings from images using a
    pretrained ResNet50 backbone (classifier head removed).
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        logger.info(f"Loading ResNet50 (pretrained) on device={self.device} ...")
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        backbone = models.resnet50(weights=weights)
        # Remove the final fully-connected classification layer
        # Keep: conv -> bn -> relu -> maxpool -> layer1-4 -> avgpool (-> 2048-dim)
        self.model = nn.Sequential(*list(backbone.children())[:-1])
        self.model.to(self.device)

        # Optionally load fine-tuned weights produced by train.py.
        # Two checkpoint formats are supported:
        #   New (Triplet Loss): has "embedding_dim" key, fc = nn.Identity() (no fc weights)
        #   Old (CrossEntropy): has "num_classes" key, fc = nn.Sequential(Dropout, Linear)
        if _MODEL_PATH.exists():
            try:
                checkpoint = torch.load(str(_MODEL_PATH), map_location=self.device,
                                        weights_only=False)
                full_model = models.resnet50(weights=None)

                if "embedding_dim" in checkpoint:
                    # New triplet-loss format — fc is Identity (no parameters)
                    full_model.fc = nn.Identity()
                    full_model.load_state_dict(checkpoint["model_state_dict"])
                    val_info = f"val_loss={checkpoint.get('val_loss', 0.0):.4f}"
                else:
                    # Old cross-entropy format — fc is Dropout + Linear
                    num_classes = checkpoint.get("num_classes", 4)
                    full_model.fc = nn.Sequential(
                        nn.Dropout(0.3),
                        nn.Linear(full_model.fc.in_features, num_classes),
                    )
                    full_model.load_state_dict(checkpoint["model_state_dict"])
                    val_info = f"val_acc={checkpoint.get('val_acc', 0.0)*100:.1f}%"

                # Strip fc — keep backbone (conv → pool → 2048-dim output)
                self.model = nn.Sequential(*list(full_model.children())[:-1])
                self.model.to(self.device)
                logger.info(
                    f"Fine-tuned ResNet50 loaded from {_MODEL_PATH} ({val_info})"
                )
            except Exception as e:
                logger.warning(
                    f"Could not load fine-tuned model ({e}). "
                    "Falling back to pretrained ImageNet weights."
                )

        self.model.eval()
        logger.info("ResNet50 backbone loaded and ready.")

    def preprocess(self, image: Union[Image.Image, str, Path]) -> torch.Tensor:
        """Load + preprocess a single image to a (1, 3, 224, 224) tensor."""
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
        return _TRANSFORM(image).unsqueeze(0)  # (1, 3, 224, 224)

    @torch.no_grad()
    def embed_single(self, image: Union[Image.Image, str, Path]) -> np.ndarray:
        """
        Embed a single image.

        Returns:
            np.ndarray of shape (2048,), L2-normalized
        """
        tensor = self.preprocess(image).to(self.device)
        feat = self.model(tensor)           # (1, 2048, 1, 1)
        feat = feat.squeeze().cpu().numpy() # (2048,)
        feat = _l2_normalize(feat)
        return feat

    @torch.no_grad()
    def embed_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed a list of image paths in batches.

        Args:
            image_paths: List of file paths.
            batch_size:  Number of images per forward pass.
            show_progress: Print a progress line every N batches.

        Returns:
            np.ndarray of shape (N, 2048), each row L2-normalized.
            Rows for failed images are zero vectors (will be skipped by indexer).
        """
        n = len(image_paths)
        embeddings = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)
        failed = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_paths = image_paths[start:end]
            tensors = []
            # Track which positions in this batch had load failures
            failed_mask = [False] * len(batch_paths)

            for j, p in enumerate(batch_paths):
                try:
                    t = self.preprocess(p)
                    tensors.append(t)
                except (UnidentifiedImageError, FileNotFoundError, OSError) as e:
                    logger.warning(f"Skipping {p}: {e}")
                    tensors.append(torch.zeros(1, 3, 224, 224))
                    failed_mask[j] = True
                    failed += 1

            batch_tensor = torch.cat(tensors, dim=0).to(self.device)  # (B, 3, 224, 224)
            feats = self.model(batch_tensor)                            # (B, 2048, 1, 1)
            feats = feats.squeeze(-1).squeeze(-1).cpu().numpy()        # (B, 2048)

            for i, feat in enumerate(feats):
                if failed_mask[i]:
                    # Explicitly zero out rows for failed images; do NOT normalize.
                    embeddings[start + i] = 0.0
                    continue
                norm = np.linalg.norm(feat)
                if norm > 1e-6:
                    embeddings[start + i] = feat / norm

            if show_progress:
                pct = int(end / n * 100)
                print(f"  Embedding progress: {end}/{n} ({pct}%)", end="\r", flush=True)

        print()  # newline after progress
        if failed:
            logger.warning(f"{failed} images failed to load and were skipped.")
        return embeddings


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return v
    return v / norm


# ── Singleton factory ─────────────────────────────────────────────────────────
_embedder_instance: ResNet50Embedder = None


def get_embedder() -> ResNet50Embedder:
    """Return a singleton embedder (loads model once per process)."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = ResNet50Embedder(device="cpu")
    return _embedder_instance
