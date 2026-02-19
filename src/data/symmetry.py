"""
src/data/symmetry.py
---------------------
Dataset for the Symmetry Detection task (GIQ benchmark).

Task:
    Given a single image of a polyhedron, classify its symmetry type.

Label scheme (multi-label — a shape can have >1 symmetry):
    Binary vector of length 3:
        [central_point_reflection, 4_fold_rotation, 5_fold_rotation]

    A shape with no recorded symmetry receives the zero vector (label=[0,0,0]).

Each __getitem__ returns:
    {
        "image":    FloatTensor [3, H, W],
        "label":    LongTensor  [3]   (multi-hot),
        "shape_id": str,
        "view":     int,
        "group":    str,
        "name":     str,
    }

For single-label classification experiments, use the `SYM_CLASSES` mapping
and access `item["label_idx"]` (dominant / first symmetry class index, or
class 0 = "none" if no symmetry).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .base import GIQBase, DEFAULT_TRANSFORM

# ---------------------------------------------------------------------------
# Symmetry taxonomy
# ---------------------------------------------------------------------------
SYM_TAGS = [
    "central point reflection",  # index 0
    "4-fold rotation",  # index 1
    "5-fold rotation",  # index 2
]
N_CLASSES = len(SYM_TAGS)  # 3 binary dimensions
SYM_TO_IDX = {s: i for i, s in enumerate(SYM_TAGS)}


def _parse_sym(sym_str: str | None) -> list[int]:
    """Convert a comma-separated symmetry string to a multi-hot vector."""
    vec = [0] * N_CLASSES
    if not sym_str or sym_str.strip().lower() == "none":
        return vec
    for part in sym_str.split(","):
        part = part.strip()
        if part in SYM_TO_IDX:
            vec[SYM_TO_IDX[part]] = 1
    return vec


class SymmetryDataset(GIQBase):
    """
    Parameters
    ----------
    split : 'train' | 'val' | 'test'
    views_per_shape : int
        How many random viewpoints to sample per shape.  Set to 20 to use
        all views (useful at inference); lower values speed up training.
    renderings_root : path to the renderings directory
    transform : torchvision transform
    seed : reproducibility seed for viewpoint sampling
    """

    def __init__(
        self,
        split: str = "train",
        views_per_shape: int = 4,
        renderings_root: str | Path | None = None,
        transform=DEFAULT_TRANSFORM,
        seed: int = 42,
    ):
        super().__init__(split, renderings_root, transform)

        import random

        rng = random.Random(seed)

        self._samples: list[dict] = []
        for sid in self.shape_ids:
            meta = self.shapes[sid]
            sym_vec = _parse_sym(meta.get("sym"))
            all_views = meta.get("viewpoints", list(range(20)))

            n = min(views_per_shape, len(all_views))
            chosen = rng.sample(all_views, n)

            for v in chosen:
                self._samples.append(
                    {
                        "shape_id": sid,
                        "view": v,
                        "group": meta.get("group", ""),
                        "name": meta.get("name", ""),
                        "sym_vec": sym_vec,
                    }
                )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self._samples[idx]
        image = self._load_image(s["shape_id"], s["view"])
        label = torch.tensor(
            s["sym_vec"], dtype=torch.float
        )  # BCEWithLogitsLoss expects float

        return {
            "image": image,
            "label": label,
            "shape_id": s["shape_id"],
            "view": s["view"],
            "group": s["group"],
            "name": s["name"],
        }

    # ------------------------------------------------------------------
    # Class helpers
    # ------------------------------------------------------------------

    @staticmethod
    def sym_tags() -> list[str]:
        return list(SYM_TAGS)

    @staticmethod
    def num_classes() -> int:
        return N_CLASSES
