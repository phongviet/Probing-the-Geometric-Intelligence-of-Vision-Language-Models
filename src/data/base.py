"""
src/data/base.py
Shared utilities and the abstract base class for all GIQ datasets.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ---------------------------------------------------------------------------
# Default image pre-processing (resize → centre-crop → ImageNet normalise)
# ---------------------------------------------------------------------------
DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Root of the repository  (src/data/base.py → ../../)
_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent.parent


def load_split(split: str) -> dict[str, Any]:
    """Return the shapes dict for *split* ('train', 'val', or 'test')."""
    path = REPO_ROOT / "data" / "giq" / "splits" / f"{split}_shapes.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}\nRun scripts/make_splits.py first."
        )
    with open(path) as f:
        return json.load(f)


def image_path(renderings_root: Path, shape_id: str, view_idx: int) -> Path:
    """
    Return the expected path to a rendered image.

    Expected directory layout (matches GIQ download):
        renderings/<group>/<shape_id>/<view_idx:04d>.jpg
    or (flat layout):
        renderings/<shape_id>/<view_idx:04d>.jpg

    We probe both and return the first that exists.
    """
    candidates = [
        renderings_root / shape_id / f"{view_idx:04d}.jpg",
        renderings_root / shape_id / f"{view_idx}.jpg",
        renderings_root / f"{shape_id}_{view_idx:04d}.jpg",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Return first candidate (caller will handle missing file gracefully)
    return candidates[0]


def get_wild_image_paths(
    renderings_root: Path, group: str, shape_id: str
) -> list[Path]:
    """
    Return a list of all wild images for a given shape_id.
    """
    wild_dir = renderings_root / "wild_images" / group / shape_id
    if not wild_dir.exists():
        return []

    images = []
    for ext in ["*.JPG", "*.jpg", "*.png", "*.JPEG", "*.jpeg"]:
        images.extend(list(wild_dir.rglob(ext)))
    return sorted(images)


class GIQBase(Dataset, ABC):
    """Abstract base — loads split metadata and exposes helpers."""

    def __init__(
        self,
        split: str,
        renderings_root: str | Path | None = None,
        transform=DEFAULT_TRANSFORM,
    ):
        self.split = split
        self.shapes = load_split(split)
        self.shape_ids = sorted(self.shapes.keys())
        self.transform = transform

        if renderings_root is None:
            renderings_root = REPO_ROOT / "data" / "giq" / "renderings"
        self.renderings_root = Path(renderings_root)

    def _load_image(self, shape_id: str, view_idx: int) -> torch.Tensor:
        """Load and transform a single rendered view."""
        path = image_path(self.renderings_root, shape_id, view_idx)
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

    def _load_wild_image(self, path: Path) -> torch.Tensor:
        """Load and transform a single wild image."""
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Any: ...
