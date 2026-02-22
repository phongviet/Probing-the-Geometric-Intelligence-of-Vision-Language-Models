from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.base import GIQBase, REPO_ROOT
from src.models.featurizers import (
    CLIPFeaturizer,
    DINOv3Featurizer,
    SigLIP2Featurizer,
)

# ---------------------------------------------------------------------------
# Dataset for Feature Extraction
# ---------------------------------------------------------------------------


class ExtractionDataset(GIQBase):
    """
    Dataset that iterates over all views of all shapes in a split.
    Returns: {"image": tensor, "shape_id": str, "view_idx": int}
    """

    def __init__(
        self,
        split: str,
        transform=None,
        output_dir: Path | None = None,
        limit: int | None = None,
    ):
        super().__init__(split=split, transform=transform)
        # We have 20 views per shape (0-19)
        self.n_views = 20
        self.samples = []

        # Filter out already processed shapes if output_dir provided
        processed_shapes = set()
        if output_dir is not None:
            # Check for existing .npz files
            # Assuming format: <output_dir>/<shape_id>.npz
            for p in output_dir.glob("*.npz"):
                processed_shapes.add(
                    p.stem
                )  # stem is filename without extension (shape_id)

        print(f"Found {len(processed_shapes)} already processed shapes. Skipping them.")

        count = 0
        for shape_id in self.shape_ids:
            if shape_id in processed_shapes:
                continue

            # Add all views for this shape
            for view_idx in range(self.n_views):
                self.samples.append((shape_id, view_idx))

            count += 1
            if limit is not None and count >= limit:
                print(f"Stopping after {count} shapes due to --limit")
                break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        shape_id, view_idx = self.samples[idx]
        img = self._load_image(shape_id, view_idx)
        return {
            "image": img,
            "shape_id": shape_id,
            "view_idx": view_idx,
        }


# ---------------------------------------------------------------------------
# Main Extraction Script
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Extract features for GIQ benchmark.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["clip", "siglip2", "dinov3"],
        help="Model architecture to use.",
    )
    parser.add_argument(
        "--model_name", type=str, default=None, help="Hugging Face model name override."
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "val", "test"],
        help="Dataset split.",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="DataLoader workers."
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use.")
    parser.add_argument(
        "--fp16", action="store_true", help="Use FP16/BF16 mixed precision."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of shapes to process (for testing).",
    )

    args = parser.parse_args()

    # 1. Setup Model
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.model == "clip":
        model_name = args.model_name or "openai/clip-vit-base-patch16"
        featurizer = CLIPFeaturizer(model_name=model_name, device=device)
    elif args.model == "siglip2":
        model_name = args.model_name or "google/siglip2-base-patch16-224"
        featurizer = SigLIP2Featurizer(model_name=model_name, device=device)
    elif args.model == "dinov3":
        model_name = args.model_name or "facebook/dinov3-vitb16-pretrain-lvd1689m"
        featurizer = DINOv3Featurizer(model_name=model_name, device=device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # 2. Setup Output Directory
    # data/giq/features/<model>/<split>
    # Usually features are organized by shape_id, so maybe just <model> root.
    # But for efficiency, maybe grouping by split is easier, or just flat shape_id folders?
    # Let's follow: experiments/features/<model_safename>/<split>/<shape_id>.npz
    # Or data/giq/features/...

    model_safe_name = model_name.replace("/", "__")
    output_dir = REPO_ROOT / "data" / "giq" / "features" / model_safe_name / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving features to: {output_dir}")

    # 3. Setup Dataset & Loader
    transform = featurizer.get_transform()
    dataset = ExtractionDataset(
        split=args.split, transform=transform, output_dir=output_dir, limit=args.limit
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(
        f"Extracting features for {len(dataset)} images ({len(dataset.shape_ids)} shapes x 20 views)..."
    )

    # 4. Inference Loop
    # We want to aggregate features per shape.
    # The dataset yields (shape_id, view_idx). The loader shuffles? No, shuffle=False.
    # Since shuffle=False and we iterate shape by shape, we can buffer features.

    # Let's use a dict buffer: keys = shape_id
    shape_buffers = {}  # shape_id -> {'global': [], 'local': [], 'views': []}

    print("Starting inference...")

    # Enable mixed precision if requested
    dtype_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if args.fp16 and "cuda" in device
        else torch.no_grad()  # fallback context manager (no-op effectively for dtype)
    )

    for batch in tqdm(loader):
        images = batch["image"].to(device)
        shape_ids = batch["shape_id"]
        view_idxs = batch["view_idx"]

        with torch.no_grad():
            with dtype_ctx:
                features = featurizer(images)

        # Move to CPU
        global_feats = features["global"].cpu().numpy()  # [B, D]
        local_feats = features["local"].cpu().numpy()  # [B, N, D]

        for i, sid in enumerate(shape_ids):
            vid = int(view_idxs[i])

            save_path = output_dir / f"{sid}.npz"

            # If the file already exists, we don't need to process it.
            # However, we've already computed the features for this batch.
            # The efficient way is to not even run inference for these, but
            # since we filter at dataset level (lines 48-56), we shouldn't see
            # already processed shapes here unless there's a race condition or partial run.
            # We'll keep the check for safety.
            if save_path.exists():
                continue

            if sid not in shape_buffers:
                shape_buffers[sid] = {"global": [], "local": [], "views": []}

            shape_buffers[sid]["global"].append(global_feats[i])
            shape_buffers[sid]["local"].append(local_feats[i])
            shape_buffers[sid]["views"].append(vid)

            # Check if we have all 20 views
            # Note: This assumes we see all 20 views. If a batch splits them,
            # the buffer persists across batches.
            if len(shape_buffers[sid]["views"]) == 20:
                # Sort by view index to ensure correct order 0..19
                # (The dataset yields them in order, but good to be safe)
                views_arr = np.array(shape_buffers[sid]["views"])
                indices = np.argsort(views_arr)

                # Stack and sort
                # global_feats[i] is [D], so stack -> [20, D]
                g_stack = np.stack(shape_buffers[sid]["global"])
                l_stack = np.stack(shape_buffers[sid]["local"])

                g_sorted = g_stack[indices]
                l_sorted = l_stack[indices]

                # Save compressed
                np.savez_compressed(
                    save_path,
                    global_features=g_sorted,
                    local_features=l_sorted,
                    shape_id=sid,
                )

                # Clear buffer for this shape
                del shape_buffers[sid]

    # Flush any remaining (shouldn't be any if all have 20 views, but for safety)
    for sid, buf in shape_buffers.items():
        if len(buf["views"]) > 0:
            print(
                f"Warning: Incomplete views for {sid} (found {len(buf['views'])}). Saving anyway."
            )
            views_arr = np.array(buf["views"])
            indices = np.argsort(views_arr)
            g_sorted = np.stack(buf["global"])[indices]
            l_sorted = np.stack(buf["local"])[indices]
            np.savez_compressed(
                output_dir / f"{sid}.npz",
                global_features=g_sorted,
                local_features=l_sorted,
                shape_id=sid,
            )


if __name__ == "__main__":
    main()
