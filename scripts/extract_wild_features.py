from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.base import GIQBase, REPO_ROOT, get_wild_image_paths
from src.models.featurizers import (
    CLIPFeaturizer,
    DINOv3Featurizer,
    SigLIP2Featurizer,
)

# ---------------------------------------------------------------------------
# Dataset for Feature Extraction
# ---------------------------------------------------------------------------


class ExtractionWildDataset(GIQBase):
    """
    Dataset that iterates over all wild images of all shapes in a split.
    Returns: {"image": tensor, "shape_id": str, "path": str}
    """

    def __init__(
        self,
        split: str,
        transform=None,
        output_dir: Path | None = None,
        limit: int | None = None,
    ):
        super().__init__(split=split, transform=transform)
        self.samples = []

        # Filter out already processed shapes if output_dir provided
        processed_shapes = set()
        if output_dir is not None:
            for p in output_dir.glob("*.npz"):
                processed_shapes.add(p.stem)

        print(
            f"Found {len(processed_shapes)} already processed wild shapes. Skipping them."
        )

        count = 0
        for shape_id in self.shape_ids:
            if shape_id in processed_shapes:
                continue

            group = self.shapes.get(shape_id, {}).get("group", "")

            # Inject "wild_images" and the group name into the path
            shape_dir = self.renderings_root / "wild_images" / group / shape_id

            wild_paths = []
            if shape_dir.exists() and shape_dir.is_dir():
                # Grab common image formats
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]:
                    wild_paths.extend(list(shape_dir.glob(ext)))

            if not wild_paths:
                continue

            for p in wild_paths:
                self.samples.append((shape_id, str(p)))

            count += 1
            if limit is not None and count >= limit:
                print(f"Stopping after {count} shapes due to --limit")
                break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        shape_id, path_str = self.samples[idx]
        img = Image.open(path_str).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return {
            "image": img,
            "shape_id": shape_id,
            "path": path_str,
        }


# ---------------------------------------------------------------------------
# Main Extraction Script
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Extract features for GIQ benchmark wild images."
    )
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
        default="train",  # Wild images are generally split-agnostic but we process them per split shapes
        choices=["train", "val", "test"],
        help="Dataset split shapes to process.",
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
    model_safe_name = model_name.replace("/", "__")
    # Save wild features in a special "wild" directory
    output_dir = REPO_ROOT / "data" / "giq" / "features" / model_safe_name / "wild"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving wild features to: {output_dir}")

    # 3. Setup Dataset & Loader
    transform = featurizer.get_transform()
    dataset = ExtractionWildDataset(
        split=args.split, transform=transform, output_dir=output_dir, limit=args.limit
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if len(dataset) == 0:
        print("No wild images found to process.")
        return

    print(f"Extracting features for {len(dataset)} wild images...")

    # 4. Inference Loop
    shape_buffers = {}  # shape_id -> {'global': [], 'local': [], 'paths': []}

    print("Starting inference...")

    dtype_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if args.fp16 and "cuda" in device
        else torch.no_grad()
    )

    for batch in tqdm(loader):
        images = batch["image"].to(device)
        shape_ids = batch["shape_id"]
        paths = batch["path"]

        with torch.no_grad():
            with dtype_ctx:
                features = featurizer(images)

        global_feats = features["global"].cpu().numpy()  # [B, D]
        local_feats = features["local"].cpu().numpy()  # [B, N, D]

        for i, sid in enumerate(shape_ids):
            path_str = paths[i]

            # Since paths can be long and absolute, let's just store the filename relative to the wild_images root
            # e.g., catalan/cid_10/indoor/DSC_0001.JPG
            # Actually, just storing the exact string the dataset yields is safest for lookup
            # but to save space, let's store relative paths if possible.
            # The dataset yields absolute paths from get_wild_image_paths.
            # We will just store the absolute paths as strings.

            save_path = output_dir / f"{sid}.npz"
            if save_path.exists():
                continue

            if sid not in shape_buffers:
                shape_buffers[sid] = {"global": [], "local": [], "paths": []}

            shape_buffers[sid]["global"].append(global_feats[i])
            shape_buffers[sid]["local"].append(local_feats[i])
            shape_buffers[sid]["paths"].append(path_str)

    # Flush all buffers
    for sid, buf in shape_buffers.items():
        if len(buf["paths"]) > 0:
            g_stack = np.stack(buf["global"])
            l_stack = np.stack(buf["local"])
            p_array = np.array(buf["paths"], dtype=str)

            np.savez_compressed(
                output_dir / f"{sid}.npz",
                global_features=g_stack,
                local_features=l_stack,
                paths=p_array,
                shape_id=sid,
            )

    print("Extraction complete.")


if __name__ == "__main__":
    main()
