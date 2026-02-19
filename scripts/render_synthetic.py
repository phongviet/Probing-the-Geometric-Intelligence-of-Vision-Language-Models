"""
Render synthetic images for the GIQ benchmark using Mitsuba 3.

Reproduces the exact setup from the GIQ paper (Section 3.3):
  - 256x256 resolution, perspective camera
  - Near clip: 1e-3, far clip: 1e8
  - Two-sided diffuse BRDF, high-reflectance yellowish surface
  - 1024 low-discrepancy samples per pixel
  - Direct integrator: 4 emitter samples, no BSDF sampling
  - 20 viewpoints uniformly sampled over the upper viewing hemisphere

Output layout (matches data loaders in src/data/):
  data/giq/renderings/<shape_id>/<view_idx:04d>.jpg

Usage:
  conda run -n geoprobe python scripts/render_synthetic.py
  conda run -n geoprobe python scripts/render_synthetic.py --shape_id wid_3
  conda run -n geoprobe python scripts/render_synthetic.py --workers 4
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
MESHES_ROOT = REPO_ROOT / "data" / "giq" / "meshes" / "3d_meshes"
RENDERINGS_ROOT = REPO_ROOT / "data" / "giq" / "renderings"
SHAPES_JSON = REPO_ROOT / "giq-benchmark" / "jsons" / "shapes.json"

# ---------------------------------------------------------------------------
# Rendering constants (from GIQ paper Section 3.3)
# ---------------------------------------------------------------------------
IMG_SIZE = 256
N_VIEWS = 20
SPP = 1024  # samples per pixel
NEAR_CLIP = 1e-3
FAR_CLIP = 1e8
EMITTER_SAMPLES = 4
# Yellowish, high-reflectance diffuse surface (approximates paper description)
SURFACE_COLOR = [0.9, 0.85, 0.5]  # warm yellow, close to paper renders
CAMERA_DISTANCE_FACTOR = 2.5  # multiplied by bounding sphere radius


# ---------------------------------------------------------------------------
# Hemisphere sampling: 20 views uniformly distributed
# ---------------------------------------------------------------------------


def fibonacci_hemisphere(n: int, seed: int = 42) -> list[tuple[float, float, float]]:
    """
    Distribute n points uniformly over the upper hemisphere using a
    Fibonacci-lattice (golden-ratio) approach.
    Returns list of (x, y, z) unit vectors with z >= 0.
    """
    rng = random.Random(seed)
    golden = (1 + math.sqrt(5)) / 2
    points = []
    for i in range(n):
        theta = math.acos(1 - (i + 0.5) / n)  # polar angle [0, pi/2]
        phi = 2 * math.pi * i / golden  # azimuthal angle
        # clamp theta to upper hemisphere
        theta = min(theta, math.pi / 2 - 0.01)
        x = math.sin(theta) * math.cos(phi)
        y = math.sin(theta) * math.sin(phi)
        z = math.cos(theta)
        points.append((x, y, z))
    return points


# ---------------------------------------------------------------------------
# OBJ bounding sphere
# ---------------------------------------------------------------------------


def compute_bounding_sphere(obj_path: Path) -> tuple[np.ndarray, float]:
    """Parse OBJ vertices and return (center, radius)."""
    verts = []
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not verts:
        return np.zeros(3), 1.0
    verts = np.array(verts)
    center = (verts.max(axis=0) + verts.min(axis=0)) / 2
    radius = float(np.linalg.norm(verts - center, axis=1).max())
    return center, max(radius, 1e-6)


# ---------------------------------------------------------------------------
# Find OBJ file for a shape id
# ---------------------------------------------------------------------------


def find_obj(shape_id: str) -> Path | None:
    for subdir in ("catalan", "johnson", "wenninger"):
        p = MESHES_ROOT / subdir / f"{shape_id}.obj"
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Build Mitsuba scene dict
# ---------------------------------------------------------------------------


def build_scene(
    obj_path: Path,
    camera_origin: tuple[float, float, float],
    center: np.ndarray,
    radius: float,
    mi,  # mitsuba module passed in to avoid re-importing
) -> dict:
    """Return a Mitsuba scene dictionary for one view."""
    ox, oy, oz = camera_origin
    cx, cy, cz = center.tolist()

    # Camera sits at direction (ox, oy, oz) * distance, looking at center
    dist = radius * CAMERA_DISTANCE_FACTOR
    eye = [cx + ox * dist, cy + oy * dist, cz + oz * dist]
    target = [cx, cy, cz]

    # Up vector: world-up (0,0,1); fall back to (0,1,0) when camera is near zenith
    view_dir = np.array(target) - np.array(eye)
    view_dir /= np.linalg.norm(view_dir)
    up = [0.0, 1.0, 0.0] if abs(np.dot(view_dir, [0, 0, 1])) > 0.95 else [0.0, 0.0, 1.0]

    # Build camera-to-world transform using Mitsuba's Python API (not dict "lookat")
    to_world = mi.ScalarTransform4f.look_at(origin=eye, target=target, up=up)

    return {
        "type": "scene",
        # Direct integrator: 4 emitter samples, no BSDF sampling (paper Sec. 3.3)
        "integrator": {
            "type": "direct",
            "emitter_samples": EMITTER_SAMPLES,
            "bsdf_samples": 0,
        },
        # Constant environment emitter — uniform ambient illumination
        "emitter": {
            "type": "constant",
            "radiance": {"type": "rgb", "value": [1.0, 1.0, 1.0]},
        },
        # Perspective camera (256×256, near=1e-3, far=1e8)
        "sensor": {
            "type": "perspective",
            "near_clip": NEAR_CLIP,
            "far_clip": FAR_CLIP,
            "fov": 45.0,
            "to_world": to_world,
            "film": {
                "type": "hdrfilm",
                "width": IMG_SIZE,
                "height": IMG_SIZE,
                "pixel_format": "rgb",
            },
            "sampler": {
                "type": "ldsampler",  # low-discrepancy, 1024 spp
                "sample_count": SPP,
            },
        },
        # Polyhedron mesh — two-sided diffuse BRDF, yellowish high-reflectance surface
        "shape": {
            "type": "obj",
            "filename": str(obj_path),
            "bsdf": {
                "type": "twosided",
                "material": {
                    "type": "diffuse",
                    "reflectance": {"type": "rgb", "value": SURFACE_COLOR},
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# Render one shape
# ---------------------------------------------------------------------------


def render_shape(shape_id: str, overwrite: bool = False) -> bool:
    import mitsuba as mi

    mi.set_variant("scalar_rgb")

    obj_path = find_obj(shape_id)
    if obj_path is None:
        print(f"  [SKIP] {shape_id}: OBJ not found", file=sys.stderr)
        return False

    out_dir = RENDERINGS_ROOT / shape_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    if not overwrite:
        existing = list(out_dir.glob("*.jpg"))
        if len(existing) >= N_VIEWS:
            print(f"  [SKIP] {shape_id}: already has {len(existing)} renders")
            return True

    center, radius = compute_bounding_sphere(obj_path)
    viewpoints = fibonacci_hemisphere(N_VIEWS)

    for view_idx, vp in enumerate(viewpoints):
        out_path = out_dir / f"{view_idx:04d}.jpg"
        if out_path.exists() and not overwrite:
            continue

        scene_dict = build_scene(obj_path, vp, center, radius, mi)

        try:
            scene = mi.load_dict(scene_dict)
            img = mi.render(scene)

            # Tone-map: sRGB gamma + clamp to uint8, save as JPEG
            bmp = mi.Bitmap(img)
            bmp = bmp.convert(
                mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, srgb_gamma=True
            )
            bmp.write(str(out_path))
        except Exception as e:
            print(f"  [ERROR] {shape_id} view {view_idx}: {e}", file=sys.stderr)
            continue

    actual = len(list(out_dir.glob("*.jpg")))
    print(f"  [DONE] {shape_id}: {actual}/{N_VIEWS} views saved")
    return actual == N_VIEWS


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Render GIQ synthetic images with Mitsuba 3"
    )
    parser.add_argument(
        "--shape_id",
        type=str,
        default=None,
        help="Render a single shape (e.g. wid_3). Renders all shapes if omitted.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-render even if output images already exist.",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of parallel worker processes."
    )
    args = parser.parse_args()

    # Load shape list
    with open(SHAPES_JSON) as f:
        shapes = json.load(f)

    if args.shape_id:
        shape_ids = [args.shape_id]
    else:
        shape_ids = list(shapes.keys())

    print(f"Rendering {len(shape_ids)} shape(s) → {RENDERINGS_ROOT}")
    print(f"  {IMG_SIZE}x{IMG_SIZE}px, {N_VIEWS} views, {SPP} spp\n")

    if args.workers > 1:
        from multiprocessing import Pool

        # Pass overwrite flag via a wrapper
        tasks = [(sid, args.overwrite) for sid in shape_ids]

        def _render(args_tuple):
            return render_shape(*args_tuple)

        with Pool(args.workers) as pool:
            results = pool.map(_render, tasks)
    else:
        results = [render_shape(sid, args.overwrite) for sid in shape_ids]

    ok = sum(results)
    print(f"\nDone: {ok}/{len(shape_ids)} shapes fully rendered.")


if __name__ == "__main__":
    main()
