"""
scripts/visualize_batch.py
--------------------------
Visualisation script to verify data loading and augmentation.

Loads batches from:
1. MentalRotationDataset (pairs)
2. SymmetryDataset (single images + labels)
3. NormalsDataset (image + normal map)

Saves a grid to experiments/data_viz.jpg.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

# Add src to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from src.data import MentalRotationDataset, SymmetryDataset, NormalsDataset  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalisation for visualisation."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean


def normals_to_rgb(normals: torch.Tensor) -> torch.Tensor:
    """Map normals [-1, 1] to [0, 1]."""
    return (normals + 1) / 2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Visualising batches...")

    # 1. Mental Rotation
    print("Loading MentalRotationDataset...")
    ds_mr = MentalRotationDataset("train")
    batch_mr = [ds_mr[i] for i in range(4)]

    # Create pairs for grid: (img_a, img_b)
    mr_imgs = []
    mr_labels = []
    for s in batch_mr:
        mr_imgs.append(denormalize(s["img_a"]))
        mr_imgs.append(denormalize(s["img_b"]))
        label_text = "SAME" if s["label"] == 1 else "DIFF"
        mr_labels.append(f"{label_text}")
        mr_labels.append("")  # Spacer for pair

    grid_mr = make_grid(mr_imgs, nrow=2, padding=2)  # 2 images per row (pair)

    # 2. Symmetry
    print("Loading SymmetryDataset...")
    ds_sym = SymmetryDataset("train")
    batch_sym = [ds_sym[i] for i in range(4)]

    sym_imgs = []
    sym_labels = []
    # Symmetry classes: ['central point reflection', '4-fold rotation', '5-fold rotation']
    sym_names = ["Cent", "4-fold", "5-fold"]

    for s in batch_sym:
        sym_imgs.append(denormalize(s["image"]))
        # Decode multi-hot
        lbls = [n for i, n in enumerate(sym_names) if s["label"][i] > 0.5]
        sym_labels.append(", ".join(lbls) if lbls else "None")

    grid_sym = make_grid(sym_imgs, nrow=4, padding=2)

    # 3. Normals
    print("Loading NormalsDataset...")
    ds_norm = NormalsDataset("train", mode="cache")
    # Try to find samples with valid normals (we only generated 5 shapes)
    # We'll just iterate until we find some valid ones or take first 4
    batch_norm = []
    valid_count = 0
    for i in range(len(ds_norm)):
        sample = ds_norm[i]
        # Check if normals are not all zero (simple check)
        if sample["normals"].abs().sum() > 0:
            batch_norm.append(sample)
            valid_count += 1
        if valid_count >= 4:
            break

    if valid_count < 4:
        print(f"Warning: Only found {valid_count} samples with valid normals.")

    norm_imgs = []
    for s in batch_norm:
        norm_imgs.append(denormalize(s["image"]))
        norm_imgs.append(normals_to_rgb(s["normals"]))

    grid_norm = make_grid(norm_imgs, nrow=2, padding=2)  # image, normal pair

    # Plotting
    plt.figure(figsize=(12, 12))

    # Subplot 1: Mental Rotation
    ax1 = plt.subplot(3, 1, 1)
    ax1.imshow(grid_mr.permute(1, 2, 0).clip(0, 1))
    ax1.set_title("Mental Rotation (Pairs)")
    ax1.axis("off")

    # Subplot 2: Symmetry
    ax2 = plt.subplot(3, 1, 2)
    ax2.imshow(grid_sym.permute(1, 2, 0).clip(0, 1))
    ax2.set_title(f"Symmetry: {sym_labels}")
    ax2.axis("off")

    # Subplot 3: Normals
    ax3 = plt.subplot(3, 1, 3)
    ax3.imshow(grid_norm.permute(1, 2, 0).clip(0, 1))
    ax3.set_title("Normals (RGB, Normal Map)")
    ax3.axis("off")

    out_dir = REPO_ROOT / "experiments"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "data_viz.jpg"
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    main()
