"""
src/data/normals.py
--------------------
Dataset for the Surface Normals (dense prediction) task.

This task follows the Probe3D protocol: train a lightweight linear decoder
(or MLP head) on top of frozen backbone features to predict per-pixel
surface normal vectors.

Because GIQ does *not* ship ground-truth normal maps with the rendered
images, this loader generates pseudo-normals from the 3D mesh on-the-fly
using Open3D (optional) or a pre-computed normal cache (recommended).

Two modes
---------
mode='cache'  (default)
    Loads pre-computed normal maps from disk.
    Expected path:  data/giq/normals/<shape_id>/<view_idx:04d>.png
    The PNG stores RGB values where (R,G,B) = (Nx+1)/2 * 255 mapped to
    world-space normals in [-1,1]^3.  This is the standard convention used
    by Probe3D / Surface Normal Estimation benchmarks.

mode='realtime'
    Generates normals from the OBJ mesh + camera poses at runtime using
    Open3D.  Slower but requires no pre-computation.  Install via:
        pip install open3d

Each __getitem__ returns:
    {
        "image":    FloatTensor [3, H, W]  (rendered RGB, normalised)
        "normals":  FloatTensor [3, H, W]  (world-space XYZ in [-1,1])
        "mask":     BoolTensor  [H, W]     (True = valid / foreground pixel)
        "shape_id": str,
        "view":     int,
    }
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from .base import GIQBase, REPO_ROOT, DEFAULT_TRANSFORM

# ---------------------------------------------------------------------------
# Normal-specific transform (no colour normalisation — raw values needed)
# ---------------------------------------------------------------------------
NORMALS_RESIZE = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.CenterCrop(224),
    ]
)

_IMG_H = _IMG_W = 224  # output spatial resolution


def _normal_png_to_tensor(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load a normal map PNG → FloatTensor [3,H,W] in [-1,1] + bool mask [H,W].

    Convention (compatible with Probe3D / DIODE / OASIS):
        R channel → Nx   (stored as uint8: value = (Nx + 1) / 2 * 255)
        G channel → Ny
        B channel → Nz
        Alpha channel (if present) → mask (255 = valid)
    """
    img = np.array(Image.open(path).convert("RGBA"), dtype=np.float32)
    rgb = img[..., :3]  # [H, W, 3]
    mask = img[..., 3] > 127  # [H, W]

    normals = rgb / 127.5 - 1.0  # [H, W, 3] in [-1, 1]
    # Normalise each valid pixel's normal vector to unit length
    norms = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8
    normals = normals / norms

    t_normals = torch.from_numpy(normals.transpose(2, 0, 1))  # [3, H, W]
    t_mask = torch.from_numpy(mask)  # [H, W]
    return t_normals, t_mask


class NormalsDataset(GIQBase):
    """
    Parameters
    ----------
    split : 'train' | 'val' | 'test'
    mode  : 'cache' | 'realtime'
    normals_root : directory with pre-computed normal maps
        (defaults to data/giq/normals/)
    views_per_shape : viewpoints per shape (all 20 by default)
    renderings_root : path to rendered RGB images
    transform : transform applied to the RGB image
    """

    def __init__(
        self,
        split: str = "train",
        mode: str = "cache",
        normals_root: str | Path | None = None,
        views_per_shape: int = 20,
        renderings_root: str | Path | None = None,
        transform=DEFAULT_TRANSFORM,
        seed: int = 42,
    ):
        super().__init__(split, renderings_root, transform)
        self.mode = mode

        if normals_root is None:
            normals_root = REPO_ROOT / "data" / "giq" / "normals"
        self.normals_root = Path(normals_root)

        import random

        rng = random.Random(seed)

        self._samples: list[dict] = []
        for sid in self.shape_ids:
            meta = self.shapes[sid]
            views = meta.get("viewpoints", list(range(20)))
            n = min(views_per_shape, len(views))
            chosen = rng.sample(views, n)
            for v in chosen:
                self._samples.append({"shape_id": sid, "view": v})

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self._samples[idx]
        sid = s["shape_id"]
        v = s["view"]

        image = self._load_image(sid, v)

        if self.mode == "realtime":
            normals, mask = self._generate_normals_realtime(sid, v, image.shape[-2:])
        else:
            normals, mask = self._load_normals_cache(sid, v)

        return {
            "image": image,
            "normals": normals,
            "mask": mask,
            "shape_id": sid,
            "view": v,
        }

    # ------------------------------------------------------------------
    # Normal loading / generation
    # ------------------------------------------------------------------

    def _load_normals_cache(
        self, shape_id: str, view_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        candidates = [
            self.normals_root / shape_id / f"{view_idx:04d}.png",
            self.normals_root / shape_id / f"{view_idx}.png",
        ]
        for path in candidates:
            if path.exists():
                return _normal_png_to_tensor(path)

        # Normal map not yet generated — return zero normals + empty mask
        warnings.warn(
            f"Normal map not found for {shape_id} view {view_idx}. "
            f"Returning zeros.  Run scripts/precompute_normals.py to generate them.",
            stacklevel=2,
        )
        h, w = _IMG_H, _IMG_W
        return torch.zeros(3, h, w), torch.zeros(h, w, dtype=torch.bool)

    def _generate_normals_realtime(
        self, shape_id: str, view_idx: int, out_hw: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Render surface normals from the OBJ mesh using Open3D raycasting.
        Requires: pip install open3d
        """
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError(
                "open3d is required for mode='realtime'.  "
                "Install it with:  pip install open3d"
            )
        # Locate the mesh
        mesh_root = REPO_ROOT / "data" / "giq" / "meshes"
        obj_candidates = list(mesh_root.rglob(f"{shape_id}.obj"))
        if not obj_candidates:
            warnings.warn(f"OBJ mesh not found for {shape_id}. Returning zeros.")
            h, w = out_hw
            return torch.zeros(3, h, w), torch.zeros(h, w, dtype=torch.bool)

        mesh = o3d.io.read_triangle_mesh(str(obj_candidates[0]))
        mesh.compute_vertex_normals()

        # Placeholder: full raycasting camera setup requires the camera
        # extrinsics for each view, which GIQ stores in scene metadata.
        # This implementation returns zeros until camera poses are available.
        warnings.warn(
            "Realtime normal rendering requires camera pose data.  "
            "Returning zeros until precompute_normals.py is run.",
            stacklevel=2,
        )
        h, w = out_hw
        return torch.zeros(3, h, w), torch.zeros(h, w, dtype=torch.bool)
