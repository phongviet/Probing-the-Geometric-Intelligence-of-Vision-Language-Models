"""
scripts/precompute_normals.py
-----------------------------
Pre-compute surface normal maps for the GIQ benchmark using Open3D.

Reads OBJ meshes, renders them from the 20 canonical viewpoints, and saves
world-space normal maps as RGBA PNGs.

Output format:
    data/giq/normals/<shape_id>/<view:04d>.png
    RGB = (normal + 1) / 2 * 255
    Alpha = mask (255 = foreground)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import open3d as o3d
except ImportError:
    print(
        "Error: open3d is required. Install with: pip install open3d", file=sys.stderr
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
MESHES_ROOT = REPO_ROOT / "data" / "giq" / "meshes"
NORMALS_ROOT = REPO_ROOT / "data" / "giq" / "normals"
SHAPES_JSON = REPO_ROOT / "giq-benchmark" / "jsons" / "shapes.json"

IMG_SIZE = 224  # Target resolution for normals (matches dataset output)
N_VIEWS = 20
CAMERA_DISTANCE_FACTOR = 2.5
FOV_DEG = 45.0


# ---------------------------------------------------------------------------
# Geometry & Camera Helpers
# ---------------------------------------------------------------------------


def compute_bounding_sphere(
    mesh: o3d.geometry.TriangleMesh,
) -> tuple[np.ndarray, float]:
    """Return (center, radius) of the mesh bounding sphere."""
    verts = np.asarray(mesh.vertices)
    if len(verts) == 0:
        return np.zeros(3), 1.0
    center = (verts.max(axis=0) + verts.min(axis=0)) / 2
    # Radius is max distance from center
    radius = float(np.linalg.norm(verts - center, axis=1).max())
    return center, max(radius, 1e-6)


def fibonacci_hemisphere(n: int) -> list[tuple[float, float, float]]:
    """
    Same sampling as render_synthetic.py.
    Returns list of (x, y, z) unit vectors.
    """
    # NOTE: No random seed here because the sequence is deterministic
    # relative to 'i' in the loop. The original script used a seed for
    # random.Random but didn't use it in the loop?
    # Wait, looking at render_synthetic.py:
    # rng = random.Random(seed) -- unused!
    # The math is deterministic.

    golden = (1 + math.sqrt(5)) / 2
    points = []
    for i in range(n):
        theta = math.acos(1 - (i + 0.5) / n)
        phi = 2 * math.pi * i / golden
        theta = min(theta, math.pi / 2 - 0.01)
        x = math.sin(theta) * math.cos(phi)
        y = math.sin(theta) * math.sin(phi)
        z = math.cos(theta)
        points.append((x, y, z))
    return points


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Compute 4x4 view matrix (world-to-camera)."""
    z_axis = eye - target
    z_axis /= np.linalg.norm(z_axis)

    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)

    # View matrix
    view = np.eye(4)
    view[:3, 0] = x_axis
    view[:3, 1] = y_axis
    view[:3, 2] = z_axis
    view[:3, 3] = -np.dot(np.vstack((x_axis, y_axis, z_axis)), eye)

    # Open3D RaycastingScene expects 'eye' to be origin in camera space?
    # Actually, RaycastingScene.CastRays takes rays.
    # We need to generate rays in world space.
    return view


def get_camera_rays(
    H: int, W: int, fov_deg: float, eye: np.ndarray, target: np.ndarray, up: np.ndarray
) -> np.ndarray:
    """
    Generate viewing rays (origin, direction) for a pinhole camera.
    Returns [H, W, 6] array where last dim is (ox, oy, oz, dx, dy, dz).
    """
    # Camera coordinate system
    # Forward: -Z (OpenGL convention)
    # Mitsuba/OpenGL: Camera looks down -Z.

    # Construct camera-to-world rotation
    z_cam = eye - target
    z_cam /= np.linalg.norm(z_cam)

    x_cam = np.cross(up, z_cam)
    x_cam /= np.linalg.norm(x_cam)

    y_cam = np.cross(z_cam, x_cam)

    # Rotation matrix (Camera -> World)
    # [x_cam, y_cam, z_cam]
    R_cw = np.stack([x_cam, y_cam, z_cam], axis=1)  # 3x3

    # Intrinsic parameters
    f = 0.5 * H / math.tan(math.radians(fov_deg) / 2)
    cx, cy = W / 2 - 0.5, H / 2 - 0.5  # Pixel centers

    # Grid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Directions in camera space (looking down -Z)
    # x = (u - cx) / f
    # y = -(v - cy) / f  (flip Y because image v goes down, cam y goes up)
    # z = -1

    x_c = (u - cx) / f
    y_c = -(v - cy) / f
    z_c = -np.ones_like(x_c)

    dirs_c = np.stack([x_c, y_c, z_c], axis=-1)  # [H, W, 3]
    dirs_c = dirs_c / np.linalg.norm(dirs_c, axis=-1, keepdims=True)

    # Transform to world space
    # dirs_w = (R_cw @ dirs_c.T).T
    # Optimize: dirs_w = dirs_c @ R_cw.T
    dirs_w = dirs_c @ R_cw.T

    # Origins are all 'eye'
    origins = np.broadcast_to(eye, dirs_w.shape)

    # Stack
    rays = np.concatenate([origins, dirs_w], axis=-1)  # [H, W, 6]
    return rays.astype(np.float32)


def process_shape(shape_id: str, overwrite: bool = False) -> None:
    # 1. Find mesh
    mesh_path = None
    # Hardcoded known structure for GIQ
    for subdir in ("catalan", "johnson", "wenninger"):
        p = MESHES_ROOT / subdir / f"{shape_id}.obj"
        if p.exists():
            mesh_path = p
            break

    if mesh_path is None:
        # Fallback search
        candidates = list(MESHES_ROOT.rglob(f"{shape_id}.obj"))
        if candidates:
            mesh_path = candidates[0]

    if mesh_path is None:
        # tqdm.write(f"Skipping {shape_id}: OBJ not found")
        return

    out_dir = NORMALS_ROOT / shape_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if done
    if not overwrite:
        existing = list(out_dir.glob("*.png"))
        if len(existing) >= N_VIEWS:
            return

    # 2. Load mesh and setup raycasting scene
    try:
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh.compute_vertex_normals()
        t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(t_mesh)
    except Exception as e:
        print(f"Error loading mesh {shape_id}: {e}", file=sys.stderr)
        return

    center, radius = compute_bounding_sphere(mesh)
    viewpoints = fibonacci_hemisphere(N_VIEWS)

    for view_idx, vp in enumerate(viewpoints):
        out_path = out_dir / f"{view_idx:04d}.png"
        if out_path.exists() and not overwrite:
            continue

        # 3. Setup Camera
        ox, oy, oz = vp
        cx, cy, cz = center
        dist = radius * CAMERA_DISTANCE_FACTOR
        eye = np.array([cx + ox * dist, cy + oy * dist, cz + oz * dist])
        target = np.array([cx, cy, cz])

        # View direction (target - eye)
        view_dir = target - eye
        view_dir_norm = np.linalg.norm(view_dir)
        if view_dir_norm < 1e-6:
            continue
        view_dir /= view_dir_norm

        # Up vector logic
        up = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(view_dir, [0, 0, 1])) > 0.95:
            up = np.array([0.0, 0.0, 1.0])

        # 4. Generate Rays
        rays = get_camera_rays(IMG_SIZE, IMG_SIZE, FOV_DEG, eye, target, up)

        # 5. Raycast
        ans = scene.cast_rays(o3d.core.Tensor(rays))

        # primitive_normals is [H, W, 3]
        # Open3D Raycasting returns 0 for misses usually?
        # We use t_hit to mask
        normals = ans["primitive_normals"].numpy()
        hit = ans["t_hit"].numpy()

        mask = np.isfinite(hit)

        # 6. Encode
        enc_normals = (normals + 1.0) / 2.0
        enc_normals = np.clip(enc_normals, 0.0, 1.0)

        img_rgb = (enc_normals * 255).astype(np.uint8)

        # Zero out background
        img_rgb[~mask] = 0

        img_rgba = np.zeros((IMG_SIZE, IMG_SIZE, 4), dtype=np.uint8)
        img_rgba[..., :3] = img_rgb
        img_rgba[..., 3] = (mask * 255).astype(np.uint8)

        Image.fromarray(img_rgba).save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not SHAPES_JSON.exists():
        # Manual fallback if shapes.json is missing
        shape_ids = [p.stem for p in MESHES_ROOT.rglob("*.obj")]
        shape_ids = sorted(list(set(shape_ids)))
    else:
        with open(SHAPES_JSON) as f:
            shapes = json.load(f)
        shape_ids = sorted(list(shapes.keys()))

    if args.subset > 0:
        shape_ids = shape_ids[: args.subset]

    print(f"Generating normals for {len(shape_ids)} shapes...")
    for sid in tqdm(shape_ids):
        process_shape(sid, args.overwrite)


if __name__ == "__main__":
    main()
