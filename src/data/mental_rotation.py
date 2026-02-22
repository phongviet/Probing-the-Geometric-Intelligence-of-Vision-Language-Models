"""
src/data/mental_rotation.py
----------------------------
Dataset for the Mental Rotation task (GIQ benchmark).

Task:
    Given two images of polyhedra (possibly different viewpoints),
    predict whether they show the *same* shape (label=1) or *different*
    shapes (label=0).

Two operating modes
-------------------
mode='all'   (default for training / validation)
    Enumerate all valid same-shape pairs (positive) and all cross-shape
    pairs (negative) up to a configurable max_negatives_per_shape cap.
    Produces a balanced dataset by under-sampling negatives.

mode='hard'  (for test evaluation)
    Uses the curated hard_examples.json pairs exactly as defined by the
    GIQ authors.  Each pair is sampled with one randomly chosen viewpoint
    per shape.

Each __getitem__ returns:
    {
        "img_a":    FloatTensor [3, H, W],
        "img_b":    FloatTensor [3, H, W],
        "label":    int  (1 = same, 0 = different),
        "shape_a":  str  (shape_id),
        "shape_b":  str  (shape_id),
        "view_a":   int,
        "view_b":   int,
    }
"""

from __future__ import annotations

import json
import random
from itertools import combinations
from pathlib import Path
from typing import Any

import torch

from .base import GIQBase, REPO_ROOT, DEFAULT_TRANSFORM, get_wild_image_paths

_HARD_JSON = REPO_ROOT / "giq-benchmark" / "jsons" / "hard_examples.json"


def _norm(sid: str) -> str:
    return sid.replace(" ", "_")


class MentalRotationDataset(GIQBase):
    """
    Parameters
    ----------
    split : 'train' | 'val' | 'test'
    mode  : 'all' | 'hard'
        'all'  — enumerate pairs from the split shapes
        'hard' — use the GIQ hard_examples.json pairs (test split only)
    max_neg_per_shape : int
        Maximum number of negative (different-shape) pairs per anchor shape
        when mode='all'.  Keeps dataset size manageable.
    renderings_root : path to the renderings directory
    transform : torchvision transform applied to every image
    seed : random seed for reproducible pair sampling
    """

    def __init__(
        self,
        split: str = "train",
        mode: str = "all",
        image_type: str = "syn-syn",
        max_neg_per_shape: int = 5,
        renderings_root: str | Path | None = None,
        transform=DEFAULT_TRANSFORM,
        seed: int = 42,
    ):
        super().__init__(split, renderings_root, transform)
        self.mode = mode
        self.image_type = image_type
        self.seed = seed
        rng = random.Random(seed)

        if mode == "hard":
            self._pairs = self._build_hard_pairs()
        else:
            self._pairs = self._build_all_pairs(rng, max_neg_per_shape)

    # ------------------------------------------------------------------
    # Pair building helpers
    # ------------------------------------------------------------------

    def _get_views(self, sid: str, is_wild: bool) -> list:
        if is_wild:
            group = self.shapes.get(sid, {}).get("group", "")
            return get_wild_image_paths(self.renderings_root, group, sid)
        return self.shapes.get(sid, {}).get("viewpoints", list(range(20)))

    def _build_all_pairs(self, rng: random.Random, max_neg: int) -> list[dict]:
        """Build positive + negative pairs from the current split shapes."""
        pairs: list[dict] = []
        ids = self.shape_ids

        is_wild_a = self.image_type == "wild-wild"
        is_wild_b = self.image_type in ["syn-wild", "wild-wild"]

        # Positive pairs: same shape, two different random viewpoints
        for sid in ids:
            views_a = self._get_views(sid, is_wild_a)
            views_b = self._get_views(sid, is_wild_b)

            if not views_a or not views_b:
                continue

            if not is_wild_b and len(views_a) < 2:
                continue

            if is_wild_b:
                v_a = rng.choice(views_a)
                v_b = rng.choice(views_b)
            else:
                v_a, v_b = rng.sample(views_a, 2)

            pairs.append(
                dict(shape_a=sid, shape_b=sid, view_a=v_a, view_b=v_b, label=1)
            )

        # Negative pairs: different shapes, one random view each
        n_pos = len(pairs)
        all_neg: list[tuple[str, str]] = list(combinations(ids, 2))
        rng.shuffle(all_neg)

        # We need to build up target negatives, but only if they have valid views
        target_neg = n_pos  # balanced dataset
        neg_pairs = []

        for sid_a, sid_b in all_neg:
            if len(neg_pairs) >= target_neg:
                break

            # Randomize order (a, b) or (b, a) to ensure both sides get a chance at being wild
            if rng.random() > 0.5:
                sid_a, sid_b = sid_b, sid_a

            views_a = self._get_views(sid_a, is_wild_a)
            views_b = self._get_views(sid_b, is_wild_b)

            if not views_a or not views_b:
                continue

            v_a = rng.choice(views_a)
            v_b = rng.choice(views_b)
            neg_pairs.append(
                dict(shape_a=sid_a, shape_b=sid_b, view_a=v_a, view_b=v_b, label=0)
            )

        pairs.extend(neg_pairs)
        rng.shuffle(pairs)
        return pairs

    def _build_hard_pairs(self) -> list[dict]:
        """Build pairs from hard_examples.json (GIQ authors' curated set)."""
        with open(_HARD_JSON) as f:
            hard = json.load(f)

        rng = random.Random(self.seed)
        pairs: list[dict] = []
        is_wild_a = self.image_type == "wild-wild"
        is_wild_b = self.image_type in ["syn-wild", "wild-wild"]

        def pick_views(sid: str, is_wild: bool):
            views = self._get_views(sid, is_wild)
            return rng.choice(views) if views else None

        for sid_a_raw, sid_b_raw in hard.get("negative", []):
            sid_a, sid_b = _norm(sid_a_raw), _norm(sid_b_raw)
            v_a = pick_views(sid_a, is_wild_a)
            v_b = pick_views(sid_b, is_wild_b)
            if v_a is not None and v_b is not None:
                pairs.append(
                    dict(
                        shape_a=sid_a,
                        shape_b=sid_b,
                        view_a=v_a,
                        view_b=v_b,
                        label=0,
                    )
                )

        for sid_a_raw, sid_b_raw in hard.get("positive", []):
            sid_a, sid_b = _norm(sid_a_raw), _norm(sid_b_raw)
            v_a = pick_views(sid_a, is_wild_a)
            v_b = pick_views(sid_b, is_wild_b)
            if v_a is not None and v_b is not None:
                pairs.append(
                    dict(
                        shape_a=sid_a,
                        shape_b=sid_b,
                        view_a=v_a,
                        view_b=v_b,
                        label=1,
                    )
                )

        return pairs

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        pair = self._pairs[idx]
        sid_a, sid_b = pair["shape_a"], pair["shape_b"]
        v_a, v_b = pair["view_a"], pair["view_b"]

        is_wild_a = isinstance(v_a, (str, Path)) and not str(v_a).isdigit()
        is_wild_b = isinstance(v_b, (str, Path)) and not str(v_b).isdigit()

        if is_wild_a:
            img_a = self._load_wild_image(Path(str(v_a)))
        else:
            img_a = self._load_image(sid_a, int(str(v_a)))  # type: ignore

        if is_wild_b:
            img_b = self._load_wild_image(Path(str(v_b)))
        else:
            img_b = self._load_image(sid_b, int(str(v_b)))  # type: ignore

        return {
            "img_a": img_a,
            "img_b": img_b,
            "label": torch.tensor(pair["label"], dtype=torch.long),
            "shape_a": sid_a,
            "shape_b": sid_b,
            "view_a": str(v_a) if isinstance(v_a, Path) else v_a,
            "view_b": str(v_b) if isinstance(v_b, Path) else v_b,
        }
