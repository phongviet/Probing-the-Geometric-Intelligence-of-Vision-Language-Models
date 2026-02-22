"""
src/data/feature_loader.py
--------------------------
Utilities for loading pre-computed features from .npz files.
Also provides a FeatureDataset wrapper mixin that replaces image loading with feature loading.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

from src.data.base import REPO_ROOT
from src.data.mental_rotation import MentalRotationDataset
from src.data.symmetry import SymmetryDataset
from src.data.normals import NormalsDataset


class FeatureLoader:
    """
    Efficiently loads pre-computed features from .npz files.

    Expected file structure:
        data/giq/features/<backbone>/<split>/<shape_id>.npz
    OR
        data/giq/features/<backbone>/<shape_id>.npz (fallback)
    """

    def __init__(self, backbone: str, split: str = "train", layer: str = "global"):
        """
        Args:
            backbone: 'clip', 'siglip2', 'dinov3', etc.
            split: 'train', 'val', 'test'
            layer: 'global' (CLS token) or 'local' (patch tokens)
        """
        self.backbone = backbone
        self.split = split
        self.layer = layer

        # Map short names to directory names
        # TODO: Unify this with extract_features.py
        self.dir_map = {
            "clip": "openai__clip-vit-base-patch16",
            "siglip2": "google__siglip2-base-patch16-224",
            "dinov3": "facebook__dinov3-vitb16-pretrain-lvd1689m",
        }

        dir_name = self.dir_map.get(backbone, backbone)

        # Determine feature root
        # Try specific split folder first
        self.root = REPO_ROOT / "data" / "giq" / "features" / dir_name / split
        if not self.root.exists():
            # Fallback to backbone root
            self.root = REPO_ROOT / "data" / "giq" / "features" / dir_name
            if not self.root.exists():
                warnings.warn(f"Feature directory not found: {self.root}")

        self._cache = {}

    def clear_cache(self):
        """Clear the internal feature cache."""
        self._cache = {}

    def has_features(self, shape_id: str) -> bool:
        """Check if features exist for a shape."""
        path = self.root / f"{shape_id}.npz"
        if path.exists():
            return True
        # Check parent/fallback
        parent_path = self.root.parent / f"{shape_id}.npz"
        if parent_path.exists():
            return True
        return False

    def get_num_views(self, shape_id: str) -> int:
        """Return the number of views available for a shape."""
        if shape_id not in self._cache:
            # We need to load it to know dimensions
            # get_features loads it into cache
            # But get_features requires a view_idx, which we don't know yet.
            # So we split the loading logic.
            self._load_to_cache(shape_id)

        return len(self._cache[shape_id])

    def _load_to_cache(self, shape_id: str):
        path = self.root / f"{shape_id}.npz"
        if not path.exists():
            parent_path = self.root.parent / f"{shape_id}.npz"
            if parent_path.exists():
                path = parent_path
            else:
                raise FileNotFoundError(f"Features not found for {shape_id}")

        data = np.load(path)

        if self.layer == "global":
            if "global_features" in data:
                feats = data["global_features"]
            elif "features" in data:
                feats = data["features"]
            else:
                raise KeyError(f"No global features found in {path}")
        elif self.layer == "local":
            if "local_features" in data:
                feats = data["local_features"]
            else:
                raise KeyError(f"No local features found in {path}")
        else:
            raise ValueError(f"Unknown layer: {self.layer}")

        self._cache[shape_id] = feats

    def get_features(self, shape_id: str, view_idx: int) -> torch.Tensor:
        """
        Load features for a specific view of a shape.
        Returns a torch Tensor.
        """
        # Load entire shape features if not cached
        if shape_id not in self._cache:
            self._load_to_cache(shape_id)

        # Retrieve specific view
        feats = self._cache[shape_id]
        if view_idx >= len(feats):
            raise IndexError(
                f"View index {view_idx} out of bounds for {shape_id} (len {len(feats)})"
            )

        return torch.from_numpy(feats[view_idx]).float()


class FeatureMentalRotationDataset(MentalRotationDataset):
    def __init__(self, *args, backbone: str = "clip", layer: str = "global", **kwargs):
        super().__init__(*args, **kwargs)
        split = kwargs.get("split", args[0] if args else "train")
        self.feature_loader = FeatureLoader(backbone=backbone, split=split, layer=layer)

        # Filter pairs to only include available shapes
        if hasattr(self, "_pairs"):
            initial_len = len(self._pairs)
            new_pairs = []
            for p in self._pairs:
                sid_a, sid_b = p["shape_a"], p["shape_b"]
                v_a, v_b = p["view_a"], p["view_b"]

                if not (
                    self.feature_loader.has_features(sid_a)
                    and self.feature_loader.has_features(sid_b)
                ):
                    continue

                try:
                    # This loads the file into cache if not present
                    n_a = self.feature_loader.get_num_views(sid_a)
                    n_b = self.feature_loader.get_num_views(sid_b)
                    if v_a < n_a and v_b < n_b:
                        new_pairs.append(p)
                except Exception:
                    continue

            self._pairs = new_pairs
            final_len = len(self._pairs)
            if final_len < initial_len:
                print(
                    f"Filtered {initial_len - final_len} pairs due to missing/incomplete features. Remaining: {final_len}"
                )

        # Clear cache to free memory and prevent pickle errors with multiprocessing
        self.feature_loader.clear_cache()

    def _load_image(self, shape_id: str, view_idx: int) -> torch.Tensor:
        """Override standard image loading to return features."""
        return self.feature_loader.get_features(shape_id, view_idx)


class FeatureSymmetryDataset(SymmetryDataset):
    def __init__(self, *args, backbone: str = "clip", layer: str = "global", **kwargs):
        super().__init__(*args, **kwargs)
        split = kwargs.get("split", args[0] if args else "train")
        self.feature_loader = FeatureLoader(backbone=backbone, split=split, layer=layer)

        # Filter samples
        if hasattr(self, "_samples"):
            initial_len = len(self._samples)
            new_samples = []
            for s in self._samples:
                sid = s["shape_id"]
                v = s["view"]

                if not self.feature_loader.has_features(sid):
                    continue

                try:
                    n = self.feature_loader.get_num_views(sid)
                    if v < n:
                        new_samples.append(s)
                except Exception:
                    continue

            self._samples = new_samples
            final_len = len(self._samples)
            if final_len < initial_len:
                print(
                    f"Filtered {initial_len - final_len} samples due to missing/incomplete features. Remaining: {final_len}"
                )

        # Clear cache to free memory
        self.feature_loader.clear_cache()

    def _load_image(self, shape_id: str, view_idx: int) -> torch.Tensor:
        """Override standard image loading to return features."""
        return self.feature_loader.get_features(shape_id, view_idx)


class FeatureNormalsDataset(NormalsDataset):
    def __init__(self, *args, backbone: str = "clip", layer: str = "local", **kwargs):
        super().__init__(*args, **kwargs)
        split = kwargs.get("split", args[0] if args else "train")
        self.feature_loader = FeatureLoader(backbone=backbone, split=split, layer=layer)

        # Filter samples
        if hasattr(self, "_samples"):
            initial_len = len(self._samples)
            new_samples = []
            for s in self._samples:
                sid = s["shape_id"]
                v = s["view"]

                if not self.feature_loader.has_features(sid):
                    continue

                try:
                    n = self.feature_loader.get_num_views(sid)
                    if v < n:
                        new_samples.append(s)
                except Exception:
                    continue

            self._samples = new_samples
            final_len = len(self._samples)
            if final_len < initial_len:
                print(
                    f"Filtered {initial_len - final_len} samples due to missing/incomplete features. Remaining: {final_len}"
                )

        # Clear cache to free memory
        self.feature_loader.clear_cache()

    def _load_image(self, shape_id: str, view_idx: int) -> torch.Tensor:
        """Override standard image loading to return features."""
        return self.feature_loader.get_features(shape_id, view_idx)
