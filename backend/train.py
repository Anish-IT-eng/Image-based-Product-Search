"""
train.py
--------
Fine-tune ResNet50 on UT-Zappos50K using Triplet Loss.

Labels : brand (p.parent.name — directory name containing the image)
         383 distinct brands in the Boots/Ankle subset.
         Within the same category/subcategory, brand is the most
         discriminative visual grouping available.

Training strategy (3 phases, v2):
  Phase 1 — Freeze backbone, train layer4 only (6 epochs, LR=1e-4)
             Batch-hard triplet loss + L2 regularization.
  Phase 2 — Unfreeze layer3 + layer4 (6 epochs, LR=5e-5).
  Phase 3 — Unfreeze layer2 + layer3 + layer4 (3 epochs, LR=1e-5).

Key improvements over v1:
  - Margin: 0.3 → 0.5 (tighter clusters, more discriminative embeddings)
  - 15 total epochs (was 5)
  - Weight decay on all optimizers (L2 regularization, prevents collapse)
  - Batch-HARD mining: hardest positive + hardest negative per anchor
    (stronger gradient signal than batch-all mining)
  - Gradient clipping (max_norm=1.0) for stable training
  - Phase 3: unfreeze layer2 for deeper feature adaptation
  - Stronger data augmentation (rotation + wider color jitter)

Best model saved to backend/data/best_model.pth when validation
triplet loss improves.

After training, re-run build_index.py to regenerate embeddings.

Usage (from project root):
    python backend/train.py
"""

import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import models, transforms

from config import settings


# ── In-process log tee: writes every line to train_log.txt AND stdout ─────────
class _TeeStream:
    """Duplicates writes to both an underlying stream and a log file."""
    def __init__(self, stream, log_path):
        self._stream = stream
        self._log    = open(log_path, "w", encoding="utf-8", buffering=1)  # line-buffered
    def write(self, data):
        self._stream.write(data)
        self._stream.flush()
        self._log.write(data)
        self._log.flush()
    def flush(self):
        self._stream.flush()
        self._log.flush()
    def fileno(self):
        return self._stream.fileno()
    def close(self):
        self._log.close()


LOG_PATH = Path(__file__).parent.parent / "train_log.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")

# ── Config ────────────────────────────────────────────────────────────────────
IMAGES_ROOT = settings.images_dir
DATA_DIR    = settings.data_dir
MODEL_OUT   = DATA_DIR / "best_model.pth"

# Batch construction: P brands × K images per brand
# IMPROVED v3: P=32, K=4 → batch=128, 4× more negatives per anchor (30→124)
# More brands in each batch = harder, more informative negatives for mining
P_BRANDS        = 32     # brands per batch (was 16)
K_PER_BRAND     = 4      # images per brand per batch (was 2) → P*K = 128
VAL_FRACTION    = 0.10   # 10% validation split
MARGIN          = 0.5    # 0.3 → 0.5 (tighter clusters)
WEIGHT_DECAY    = 1e-4   # L2 regularization
EPOCHS_PHASE1   = 10     # layer4 only (was 6) — best val at ep2, need more Phase1 time
EPOCHS_PHASE2   = 6      # layer3 + layer4
EPOCHS_PHASE3   = 3      # layer2 + layer3 + layer4
LR_PHASE1       = 1e-4
LR_PHASE2       = 5e-5
LR_PHASE3       = 1e-5
MIN_IMGS_BRAND  = 2      # minimum images a brand must have

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# IMPROVED: stronger augmentation
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.55, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.25, hue=0.08),
    transforms.RandomGrayscale(p=0.08),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


# ── Dataset ───────────────────────────────────────────────────────────────────

class BrandDataset(Dataset):
    """Each item: (image_tensor, brand_idx)."""

    def __init__(self, image_paths, brand_ids, transform=None):
        self.image_paths = image_paths
        self.brand_ids   = brand_ids
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path  = self.image_paths[idx]
        label = self.brand_ids[idx]
        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, label


class BalancedBrandSampler(Sampler):
    """
    At each iteration yields P*K indices: P brands, K images each.
    Ensures each batch has many valid triplets for mining.
    """

    def __init__(self, brand_ids, p_brands, k_per_brand, seed=42):
        self.brand_to_indices = defaultdict(list)
        for i, b in enumerate(brand_ids):
            self.brand_to_indices[b].append(i)
        self.valid_brands = [
            b for b, idxs in self.brand_to_indices.items()
            if len(idxs) >= 2
        ]
        self.p_brands    = p_brands
        self.k_per_brand = k_per_brand
        self.rng         = np.random.default_rng(seed)
        self.n_batches   = max(1, len(self.valid_brands) // p_brands)

    def __len__(self):
        return self.n_batches * self.p_brands * self.k_per_brand

    def __iter__(self):
        rng = np.random.default_rng(self.rng.integers(1 << 31))
        brands = rng.permutation(self.valid_brands).tolist()
        for i in range(0, len(brands) - self.p_brands + 1, self.p_brands):
            batch_brands = brands[i : i + self.p_brands]
            for b in batch_brands:
                pool   = self.brand_to_indices[b]
                chosen = rng.choice(pool, size=min(self.k_per_brand, len(pool)), replace=False)
                if len(chosen) < self.k_per_brand:
                    extra  = rng.choice(pool, size=self.k_per_brand - len(chosen), replace=True)
                    chosen = np.concatenate([chosen, extra])
                for idx in chosen:
                    yield int(idx)


# ── Triplet Loss — Batch Hard Mining ─────────────────────────────────────────

def batch_hard_triplet_loss(embeddings, labels, margin=0.5):
    """
    IMPROVED: Batch-Hard mining — for each anchor, find the hardest positive
    (furthest same-class) and hardest negative (closest different-class).
    More efficient and produces stronger gradients than batch-all mining.

    embeddings : (N, D) L2-normalised
    labels     : (N,) integer class IDs
    Returns    : scalar loss
    """
    n = embeddings.size(0)
    # Pairwise cosine distances (1 - cosine_similarity for L2-normalised vecs)
    dot  = torch.mm(embeddings, embeddings.t())  # (N, N)
    dist = 1.0 - dot                             # cosine distance in [0, 2]

    labels_col = labels.unsqueeze(1)
    labels_row = labels.unsqueeze(0)
    pos_mask = (labels_col == labels_row)
    neg_mask = ~pos_mask
    eye      = torch.eye(n, device=embeddings.device, dtype=torch.bool)

    # Hardest positive: furthest same-class pair for each anchor
    pos_only = dist.clone()
    pos_only[~pos_mask | eye] = -1.0  # mask out non-positives
    hardest_pos, _ = pos_only.max(dim=1)  # (N,)

    # Hardest negative: closest different-class pair for each anchor
    neg_only = dist.clone()
    neg_only[~neg_mask] = 2.0  # mask out non-negatives with max distance
    hardest_neg, _ = neg_only.min(dim=1)  # (N,)

    triplet_loss = F.relu(hardest_pos - hardest_neg + margin)

    # Only count anchors that have at least one valid positive and one valid negative
    has_pos = (pos_mask & ~eye).any(dim=1)
    has_neg = neg_mask.any(dim=1)
    valid   = has_pos & has_neg

    if valid.sum() == 0:
        return triplet_loss.sum() * 0.0

    loss = triplet_loss[valid].mean()
    return loss


def batch_all_triplet_loss(embeddings, labels, margin=0.5):
    """
    Fallback: mine ALL valid triplets from a batch and compute mean loss.
    Used for validation since it gives a stable loss value across all triplets.
    """
    n = embeddings.size(0)
    dot  = torch.mm(embeddings, embeddings.t())
    dist = 1.0 - dot

    labels_col = labels.unsqueeze(1)
    labels_row = labels.unsqueeze(0)
    pos_mask = (labels_col == labels_row)
    neg_mask = ~pos_mask
    eye = torch.eye(n, device=embeddings.device, dtype=torch.bool)

    ap = dist.unsqueeze(2)
    an = dist.unsqueeze(1)
    loss_triplets = F.relu(ap - an + margin)

    pos_valid  = pos_mask & ~eye
    neg_valid  = neg_mask
    valid_mask = pos_valid.unsqueeze(2) & neg_valid.unsqueeze(1)

    loss_triplets = loss_triplets * valid_mask.float()
    active = (loss_triplets > 1e-6) & valid_mask
    n_active = active.sum().item()

    if n_active == 0:
        return loss_triplets.sum() * 0.0, 0

    loss = loss_triplets.sum() / (n_active + 1e-8)
    return loss, n_active


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model():
    """ResNet50 backbone with FC head replaced by Identity."""
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    model   = models.resnet50(weights=weights)
    model.fc = nn.Identity()
    for param in model.parameters():
        param.requires_grad = False
    return model.to(DEVICE)


def unfreeze_layers(model, layer_names):
    for name in layer_names:
        for param in getattr(model, name).parameters():
            param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Unfrozen: {layer_names}  |  Trainable params: {trainable:,}")


def get_embedding(model, imgs):
    """Forward pass → L2-normalised 2048-dim embedding."""
    feats = model(imgs)  # (N, 2048)
    return F.normalize(feats, p=2, dim=1)


# ── Training loop ─────────────────────────────────────────────────────────────

def run_train_epoch(model, loader, optimizer):
    model.train()
    total_loss, n_batches = 0.0, 0

    for imgs, labels in loader:
        imgs   = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        embeds = get_embedding(model, imgs)
        loss   = batch_hard_triplet_loss(embeds, labels, margin=MARGIN)

        if loss.requires_grad:
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


def run_val_epoch(model, loader):
    model.eval()
    total_loss, total_active, n_batches = 0.0, 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            embeds = get_embedding(model, imgs)
            loss, n_active = batch_all_triplet_loss(embeds, labels, margin=MARGIN)
            total_loss   += loss.item()
            total_active += n_active
            n_batches    += 1

    return total_loss / max(n_batches, 1), total_active


# ── Data collection ───────────────────────────────────────────────────────────

def collect_data():
    if not IMAGES_ROOT.exists():
        raise FileNotFoundError(
            f"Images directory not found: {IMAGES_ROOT}\n"
            "Extract ut-zap50k-images.zip first."
        )

    # Use brand (parent directory) as the label
    # Directory structure: Category/SubCategory/Brand/image.jpg
    brand_to_paths = defaultdict(list)
    for p in sorted(IMAGES_ROOT.rglob("*.jpg")):
        brand = p.parent.name
        brand_to_paths[brand].append(p)

    # Keep only brands with enough images to form at least one triplet
    brand_to_paths = {b: ps for b, ps in brand_to_paths.items()
                      if len(ps) >= MIN_IMGS_BRAND}

    brand_names = sorted(brand_to_paths.keys())
    brand2idx   = {b: i for i, b in enumerate(brand_names)}

    all_paths, all_labels = [], []
    for brand, paths in brand_to_paths.items():
        for p in paths:
            all_paths.append(p)
            all_labels.append(brand2idx[brand])

    return all_paths, all_labels, brand2idx


def split_data(paths, labels, val_fraction, seed=42):
    """Stratified split: same brand distribution in train and val."""
    rng = np.random.default_rng(seed)
    brand_to_indices = defaultdict(list)
    for i, lbl in enumerate(labels):
        brand_to_indices[lbl].append(i)

    train_idx, val_idx = [], []
    for lbl, idxs in brand_to_indices.items():
        idxs  = rng.permutation(idxs).tolist()
        n_val = max(1, int(len(idxs) * val_fraction)) if len(idxs) >= 2 else 0
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    return train_idx, val_idx


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Tee all output to train_log.txt directly from Python (bypasses shell buffering)
    sys.stdout = _TeeStream(sys.__stdout__, str(LOG_PATH))

    total_epochs = EPOCHS_PHASE1 + EPOCHS_PHASE2 + EPOCHS_PHASE3
    t_start = time.time()
    print(f"\n{'='*60}", flush=True)
    print(f"  VisualFind -- ResNet50 Triplet-Loss Fine-Tuning v3", flush=True)
    print(f"  Device: {DEVICE}  |  Margin: {MARGIN}  |  Total epochs: {total_epochs}", flush=True)
    print(f"  Labels: brand  |  Mining: batch-hard  |  P={P_BRANDS} K={K_PER_BRAND}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # 1. Collect data
    print("Step 1/6  Collecting image paths ...", flush=True)
    all_paths, all_labels, brand2idx = collect_data()
    n_brands = len(brand2idx)
    print(f"          -> {len(all_paths)} images | {n_brands} brands\n", flush=True)

    if len(all_paths) == 0:
        print("ERROR: No images found.")
        sys.exit(1)

    # 2. Split
    print("Step 2/6  Stratified train/val split ...", flush=True)
    train_idx, val_idx = split_data(all_paths, all_labels, VAL_FRACTION)
    train_paths  = [all_paths[i]  for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    val_paths    = [all_paths[i]  for i in val_idx]
    val_labels   = [all_labels[i] for i in val_idx]
    print(f"          -> train: {len(train_paths)} | val: {len(val_paths)}\n", flush=True)

    train_ds  = BrandDataset(train_paths, train_labels, transform=TRAIN_TRANSFORM)
    val_ds    = BrandDataset(val_paths,   val_labels,   transform=VAL_TRANSFORM)
    sampler   = BalancedBrandSampler(train_labels, P_BRANDS, K_PER_BRAND)

    train_loader = DataLoader(train_ds, batch_size=P_BRANDS * K_PER_BRAND,
                              sampler=sampler, num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    # 3. Build model
    print("Step 3/6  Building ResNet50 (backbone frozen) ...", flush=True)
    model = build_model()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    print(f"          -> ResNet50 ready, 2048-dim pool embeddings\n", flush=True)

    # ── Phase 1: layer4 ───────────────────────────────────────────────────────
    print(f"Step 4/6  Phase 1 -- layer4 only ({EPOCHS_PHASE1} epochs, LR={LR_PHASE1}) ...", flush=True)
    unfreeze_layers(model, ["layer4"])
    opt1   = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=LR_PHASE1, weight_decay=WEIGHT_DECAY)
    sched1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=EPOCHS_PHASE1, eta_min=LR_PHASE1 * 0.1)

    for ep in range(1, EPOCHS_PHASE1 + 1):
        t0 = time.time()
        tl = run_train_epoch(model, train_loader, opt1)
        vl, va = run_val_epoch(model, val_loader)
        sched1.step()  # step after optimizer
        elapsed = time.time() - t0
        print(f"  Ep {ep:>2}/{EPOCHS_PHASE1}  train_loss={tl:.4f}  val_loss={vl:.4f}  "
              f"active_triplets={va}  [{elapsed:.0f}s]", flush=True)
        if vl < best_val_loss:
            best_val_loss = vl
            torch.save({
                "epoch": ep,
                "phase": 1,
                "model_state_dict": model.state_dict(),
                "val_loss": vl,
                "brand2idx": brand2idx,
                "num_brands": n_brands,
                "embedding_dim": 2048,
                "margin": MARGIN,
                "label_type": "brand",
            }, MODEL_OUT)
            print(f"    [SAVED] best val_loss={vl:.4f}", flush=True)

    # ── Phase 2: layer3 + layer4 ──────────────────────────────────────────────
    print(f"\nStep 5/6  Phase 2 -- layer3+layer4 ({EPOCHS_PHASE2} epochs, LR={LR_PHASE2}) ...", flush=True)
    unfreeze_layers(model, ["layer3", "layer4"])
    opt2   = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=LR_PHASE2, weight_decay=WEIGHT_DECAY)
    sched2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=EPOCHS_PHASE2, eta_min=LR_PHASE2 * 0.1)

    for ep in range(1, EPOCHS_PHASE2 + 1):
        t0 = time.time()
        tl = run_train_epoch(model, train_loader, opt2)
        vl, va = run_val_epoch(model, val_loader)
        sched2.step()  # step after optimizer
        elapsed = time.time() - t0
        print(f"  Ep {ep:>2}/{EPOCHS_PHASE2}  train_loss={tl:.4f}  val_loss={vl:.4f}  "
              f"active_triplets={va}  [{elapsed:.0f}s]", flush=True)
        if vl < best_val_loss:
            best_val_loss = vl
            torch.save({
                "epoch": EPOCHS_PHASE1 + ep,
                "phase": 2,
                "model_state_dict": model.state_dict(),
                "val_loss": vl,
                "brand2idx": brand2idx,
                "num_brands": n_brands,
                "embedding_dim": 2048,
                "margin": MARGIN,
                "label_type": "brand",
            }, MODEL_OUT)
            print(f"    [SAVED] best val_loss={vl:.4f}", flush=True)

    # ── Phase 3: layer2 + layer3 + layer4 ────────────────────────────────────
    print(f"\nStep 6/6  Phase 3 -- layer2+layer3+layer4 ({EPOCHS_PHASE3} epochs, LR={LR_PHASE3}) ...", flush=True)
    unfreeze_layers(model, ["layer2", "layer3", "layer4"])
    opt3   = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=LR_PHASE3, weight_decay=WEIGHT_DECAY)
    sched3 = optim.lr_scheduler.CosineAnnealingLR(opt3, T_max=EPOCHS_PHASE3, eta_min=LR_PHASE3 * 0.1)

    for ep in range(1, EPOCHS_PHASE3 + 1):
        t0 = time.time()
        tl = run_train_epoch(model, train_loader, opt3)
        vl, va = run_val_epoch(model, val_loader)
        sched3.step()  # step after optimizer
        elapsed = time.time() - t0
        print(f"  Ep {ep:>2}/{EPOCHS_PHASE3}  train_loss={tl:.4f}  val_loss={vl:.4f}  "
              f"active_triplets={va}  [{elapsed:.0f}s]", flush=True)
        if vl < best_val_loss:
            best_val_loss = vl
            torch.save({
                "epoch": EPOCHS_PHASE1 + EPOCHS_PHASE2 + ep,
                "phase": 3,
                "model_state_dict": model.state_dict(),
                "val_loss": vl,
                "brand2idx": brand2idx,
                "num_brands": n_brands,
                "embedding_dim": 2048,
                "margin": MARGIN,
                "label_type": "brand",
            }, MODEL_OUT)
            print(f"    [SAVED] best val_loss={vl:.4f}", flush=True)

    total = time.time() - t_start
    print(f"\n{'='*60}", flush=True)
    print(f"  Training complete in {total/60:.1f} min", flush=True)
    print(f"  Best val_loss : {best_val_loss:.4f}", flush=True)
    print(f"  Model saved   : {MODEL_OUT}", flush=True)
    print(f"\n  Next step: python backend/build_index.py", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()
