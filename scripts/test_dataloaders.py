"""
scripts/test_dataloaders.py
----------------------------
Smoke-test all three dataset classes WITHOUT requiring the 90 GB renderings.
Creates tiny dummy images on-the-fly so the test works immediately.
"""

import sys
import json
import tempfile
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import MentalRotationDataset, SymmetryDataset, NormalsDataset

# --- create dummy renderings directory -----------------------------------
shapes_path = ROOT / "giq-benchmark" / "jsons" / "shapes.json"
splits_dir = ROOT / "data" / "giq" / "splits"

with open(shapes_path) as f:
    all_shapes = json.load(f)

DUMMY_DIR = Path(tempfile.mkdtemp(prefix="giq_dummy_"))
print(f"Creating dummy images in {DUMMY_DIR}")
N_VIEWS = 20
for sid in all_shapes.keys():  # all shapes
    (DUMMY_DIR / sid).mkdir(exist_ok=True)
    for v in range(N_VIEWS):
        arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        Image.fromarray(arr).save(DUMMY_DIR / sid / f"{v:04d}.jpg")


# 1. Mental Rotation
# -------------------------------------------------------------------------
print("\n--- MentalRotationDataset (train, mode=all) ---")
ds_mr = MentalRotationDataset(
    split="train",
    mode="all",
    renderings_root=DUMMY_DIR,
    max_neg_per_shape=2,
)
print(f"  Dataset size: {len(ds_mr)}")
sample = ds_mr[0]
print(f"  img_a shape: {sample['img_a'].shape}")
print(f"  img_b shape: {sample['img_b'].shape}")
print(
    f"  label: {sample['label'].item()}  ({sample['shape_a']} vs {sample['shape_b']})"
)

loader_mr = DataLoader(ds_mr, batch_size=4, shuffle=True)
batch = next(iter(loader_mr))
print(f"  Batch img_a: {batch['img_a'].shape}  labels: {batch['label'].tolist()}")

# -------------------------------------------------------------------------
# 2. Symmetry
# -------------------------------------------------------------------------
print("\n--- SymmetryDataset (train) ---")
ds_sym = SymmetryDataset(
    split="train",
    views_per_shape=2,
    renderings_root=DUMMY_DIR,
)
print(f"  Dataset size: {len(ds_sym)}")
sample = ds_sym[0]
print(f"  image shape: {sample['image'].shape}")
print(f"  label (multi-hot): {sample['label'].tolist()}  shape: {sample['shape_id']}")
print(f"  Symmetry tags: {SymmetryDataset.sym_tags()}")

loader_sym = DataLoader(ds_sym, batch_size=4, shuffle=True)
batch = next(iter(loader_sym))
print(f"  Batch images: {batch['image'].shape}  labels: {batch['label'].shape}")

# -------------------------------------------------------------------------
# 3. Normals
# -------------------------------------------------------------------------
print("\n--- NormalsDataset (train, mode=cache — no maps available) ---")
ds_norm = NormalsDataset(
    split="train",
    mode="cache",
    views_per_shape=2,
    renderings_root=DUMMY_DIR,
    normals_root=DUMMY_DIR / "_normals_nonexistent",
)
print(f"  Dataset size: {len(ds_norm)}")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sample = ds_norm[0]
print(f"  image shape:   {sample['image'].shape}")
print(f"  normals shape: {sample['normals'].shape}  (zeros — no cache yet)")
print(f"  mask shape:    {sample['mask'].shape}")

loader_norm = DataLoader(ds_norm, batch_size=4, shuffle=False)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    batch = next(iter(loader_norm))
print(f"  Batch images: {batch['image'].shape}  normals: {batch['normals'].shape}")

# -------------------------------------------------------------------------
# Mental Rotation — hard mode (test split)
# -------------------------------------------------------------------------
print("\n--- MentalRotationDataset (test, mode=hard) ---")
# Ensure dummy images exist for all hard-pair shapes
hard_path = ROOT / "giq-benchmark" / "jsons" / "hard_examples.json"
with open(hard_path) as f:
    hard = json.load(f)
hard_sids = set()
for pairs in hard.values():
    for p in pairs:
        for s in p:
            hard_sids.add(s.replace(" ", "_"))
for sid in hard_sids:
    (DUMMY_DIR / sid).mkdir(exist_ok=True)
    for v in range(N_VIEWS):
        img_path = DUMMY_DIR / sid / f"{v:04d}.jpg"
        if not img_path.exists():
            arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            Image.fromarray(arr).save(img_path)

ds_hard = MentalRotationDataset(
    split="test",
    mode="hard",
    renderings_root=DUMMY_DIR,
)
print(f"  Hard-pair dataset size: {len(ds_hard)}")
s = ds_hard[0]
print(f"  label={s['label'].item()}  {s['shape_a']} vs {s['shape_b']}")

print("\nAll smoke tests passed.")
