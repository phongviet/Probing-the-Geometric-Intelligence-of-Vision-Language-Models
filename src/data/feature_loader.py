"""
src/data/feature_loader.py
--------------------------
Utilities for loading pre-computed features from .npz files.
Also provides a FeatureDataset wrapper mixin that replaces image loading with feature loading.
"""

from __future__ import annotations

import warnings
from pathlib import Path

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

    def has_features(self, shape_id: str, is_wild: bool = False) -> bool:
        """Check if features exist for a shape."""
        if is_wild:
            path = self.root.parent / "wild" / f"{shape_id}.npz"
            return path.exists()

        path = self.root / f"{shape_id}.npz"
        if path.exists():
            return True
        # Check parent/fallback
        parent_path = self.root.parent / f"{shape_id}.npz"
        if parent_path.exists():
            return True
        return False

    def get_num_views(self, shape_id: str, is_wild: bool = False) -> int:
        """Return the number of views available for a shape."""
        cache_key = f"{shape_id}_wild" if is_wild else shape_id
        if cache_key not in self._cache:
            self._load_to_cache(shape_id, is_wild)

        if is_wild:
            return len(self._cache[cache_key]["features"])
        return len(self._cache[cache_key])

    def _load_to_cache(self, shape_id: str, is_wild: bool = False):
        if is_wild:
            path = self.root.parent / "wild" / f"{shape_id}.npz"
            if not path.exists():
                raise FileNotFoundError(f"Wild features not found for {shape_id}")
        else:
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

        cache_key = f"{shape_id}_wild" if is_wild else shape_id

        if is_wild:
            if "paths" not in data:
                raise KeyError(f"No paths array found in wild feature file {path}")
            # Store both features and paths map
            paths_list = data["paths"].tolist()
            # Only use the filename for the index lookup to avoid absolute vs relative path mismatch issues
            path_to_idx = {Path(p).name: i for i, p in enumerate(paths_list)}
            self._cache[cache_key] = {"features": feats, "path_to_idx": path_to_idx}
        else:
            self._cache[cache_key] = feats

    def get_features(self, shape_id: str, view_idx: int | str) -> torch.Tensor:
        """
        Load features for a specific view of a shape.
        view_idx is an int for synthetic views, or a string path for wild views.
        Returns a torch Tensor.
        """
        is_wild = isinstance(view_idx, str)
        cache_key = f"{shape_id}_wild" if is_wild else shape_id

        # Load entire shape features if not cached
        if cache_key not in self._cache:
            self._load_to_cache(shape_id, is_wild)

        if is_wild:
            feats = self._cache[cache_key]["features"]
            path_to_idx = self._cache[cache_key]["path_to_idx"]

            # Use filename for comparison
            filename = Path(view_idx).name
            if filename not in path_to_idx:
                raise KeyError(
                    f"Path {filename} not found in extracted wild features for {shape_id}"
                )

            idx = path_to_idx[filename]
            return torch.from_numpy(feats[idx]).float()
        else:
            # Retrieve specific view
            feats = self._cache[cache_key]
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

                is_wild_a = (
                    isinstance(v_a, (str, Path))
                    and not str(v_a).isdigit()
                    and (
                        "/" in str(v_a)
                        or "\\" in str(v_a)
                        or ".jpg" in str(v_a).lower()
                        or ".png" in str(v_a).lower()
                    )
                )
                is_wild_b = (
                    isinstance(v_b, (str, Path))
                    and not str(v_b).isdigit()
                    and (
                        "/" in str(v_b)
                        or "\\" in str(v_b)
                        or ".jpg" in str(v_b).lower()
                        or ".png" in str(v_b).lower()
                    )
                )

                if not (
                    self.feature_loader.has_features(sid_a, is_wild_a)
                    and self.feature_loader.has_features(sid_b, is_wild_b)
                ):
                    continue

                try:
                    # For wild, we don't strictly check if view < n because wild paths can just be checked for existence
                    # Or we just let feature loader raise an error during runtime if missing, but it's better to check
                    if not is_wild_a:
                        n_a = self.feature_loader.get_num_views(sid_a, is_wild_a)
                        if int(v_a) >= n_a:
                            continue

                    if not is_wild_b:
                        n_b = self.feature_loader.get_num_views(sid_b, is_wild_b)
                        if int(v_b) >= n_b:
                            continue

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

    def _load_wild_image(self, path: Path | str) -> torch.Tensor:
        """Override standard wild image loading to return features."""
        # Find shape_id from path... Wait! We don't have shape_id here easily.
        # But we can extract it from the path. Wild paths are typically: .../catalan/cid_10/indoor/...
        # Alternatively, we can let MentalRotationDataset just call get_features directly since it knows the shape_id.
        path_str = str(path)
        # Find shape_id by looking for parts of the path
        parts = Path(path_str).parts
        shape_id = None
        for p in parts:
            if "id_" in p:
                shape_id = p
                break

        if not shape_id:
            raise ValueError(f"Could not extract shape_id from wild path: {path_str}")

        return self.feature_loader.get_features(shape_id, path_str)


class FeatureSymmetryDataset(SymmetryDataset):
    def __init__(self, *args, backbone: str = "clip", layer: str = "global", **kwargs):
        super().__init__(*args, **kwargs)
        split = kwargs.get("split", args[0] if args else "train")
        self.feature_loader = FeatureLoader(backbone=backbone, split=split, layer=layer)
        self.is_wild = kwargs.get("image_type", "synthetic") == "wild"

        # Filter samples
        if hasattr(self, "_samples"):
            initial_len = len(self._samples)
            new_samples = []
            for s in self._samples:
                sid = s["shape_id"]
                v = s["view"]

                if not self.feature_loader.has_features(sid, self.is_wild):
                    continue

                try:
                    if not self.is_wild:
                        n = self.feature_loader.get_num_views(sid, self.is_wild)
                        if v < n:
                            new_samples.append(s)
                    else:
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

    def _load_wild_image(self, path: Path | str) -> torch.Tensor:
        """Override standard wild image loading to return features."""
        path_str = str(path)
        parts = Path(path_str).parts
        shape_id = None
        for p in parts:
            if "id_" in p:
                shape_id = p
                break

        if not shape_id:
            raise ValueError(f"Could not extract shape_id from wild path: {path_str}")

        return self.feature_loader.get_features(shape_id, path_str)


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
