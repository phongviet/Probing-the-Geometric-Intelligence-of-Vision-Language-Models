"""
scripts/evaluate.py
-------------------
Evaluate trained probes on the test set and compute rigorous metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.feature_loader import (
    FeatureMentalRotationDataset,
    FeatureSymmetryDataset,
    FeatureNormalsDataset,
)
from src.models.probes import LinearProbe, MLPProbe, DenseProbe


def get_dataset(task, backbone, layer, split, **kwargs):
    if task == "rotation":
        return FeatureMentalRotationDataset(
            split=split, backbone=backbone, layer=layer, mode="all", **kwargs
        )
    elif task == "symmetry":
        return FeatureSymmetryDataset(
            split=split, backbone=backbone, layer=layer, **kwargs
        )
    elif task == "normals":
        return FeatureNormalsDataset(
            split=split, backbone=backbone, layer=layer, **kwargs
        )
    else:
        raise ValueError(f"Unknown task: {task}")


def compute_angular_error(pred, gt, mask=None):
    """
    Compute angular error between predicted and ground truth normals.
    pred: [B, 3, H, W]
    gt: [B, 3, H, W]
    mask: [B, 1, H, W] (optional)
    Returns: Mean angular error in degrees.
    """
    # Normalize
    pred = torch.nn.functional.normalize(pred, dim=1)
    gt = torch.nn.functional.normalize(gt, dim=1)

    # Dot product clamped to [-1, 1]
    dot = torch.sum(pred * gt, dim=1)
    dot = torch.clamp(dot, -1.0, 1.0)

    # Arccos to get angle in radians
    angles = torch.acos(dot)

    # Convert to degrees
    angles_deg = angles * (180.0 / np.pi)

    if mask is not None:
        mask = mask.squeeze(1).bool()
        valid_angles = angles_deg[mask]
    else:
        valid_angles = angles_deg

    return valid_angles.mean().item(), torch.sqrt((valid_angles**2).mean()).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", type=str, required=True, choices=["rotation", "symmetry", "normals"]
    )
    parser.add_argument(
        "--backbone", type=str, required=True, choices=["clip", "siglip2", "dinov3"]
    )
    parser.add_argument(
        "--probe", type=str, default="linear", choices=["linear", "mlp", "dense"]
    )
    parser.add_argument(
        "--layer", type=str, default="global", choices=["global", "local"]
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--combine_method",
        type=str,
        default="concat",
        choices=["concat", "diff", "mult"],
        help="For rotation task",
    )
    parser.add_argument("--model_dir", type=str, default="experiments/probes")
    parser.add_argument("--output_file", type=str, default="experiments/results.json")
    args = parser.parse_args()

    # Load Dataset (Test Split)
    print(
        f"Loading {args.task} test dataset with {args.backbone} features ({args.layer})..."
    )
    test_ds = get_dataset(args.task, args.backbone, args.layer, "test")
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Infer input dimension
    sample = test_ds[0]
    if args.task == "rotation":
        feat_dim = sample["img_a"].shape[-1]
        if args.combine_method == "concat":
            input_dim = feat_dim * 2
        else:
            input_dim = feat_dim
        num_classes = 2
    elif args.task == "symmetry":
        feat_dim = sample["image"].shape[-1]
        input_dim = feat_dim
        num_classes = 3
    elif args.task == "normals":
        feat_dim = sample["image"].shape[-1]
        input_dim = feat_dim
        num_classes = 3
    else:
        raise ValueError(f"Unknown task: {args.task}")

    print(f"Input dimension: {input_dim}")

    # Initialize Model
    if args.probe == "linear":
        model = LinearProbe(input_dim, num_classes)
    elif args.probe == "mlp":
        model = MLPProbe(input_dim, hidden_dim=input_dim, num_classes=num_classes)
    elif args.probe == "dense":
        if args.task != "normals":
            raise ValueError("Dense probe only for normals task")
        model = DenseProbe(input_dim, num_classes)
    else:
        raise ValueError(f"Unknown probe: {args.probe}")

    model.to(args.device)

    # Load Checkpoint
    checkpoint_path = (
        Path(args.model_dir)
        / f"{args.task}_{args.backbone}_{args.probe}"
        / "best_model.pth"
    )
    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading checkpoint from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=args.device))
    model.eval()

    # Evaluation Loop
    all_preds = []
    all_labels = []
    total_angular_error = 0.0
    total_rmse_error = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            if args.task == "rotation":
                f_a = batch["img_a"].to(args.device)
                f_b = batch["img_b"].to(args.device)
                labels = batch["label"].to(args.device)

                if args.combine_method == "concat":
                    x = torch.cat([f_a, f_b], dim=-1)
                elif args.combine_method == "diff":
                    x = torch.abs(f_a - f_b)
                elif args.combine_method == "mult":
                    x = f_a * f_b

                logits = model(x)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            elif args.task == "symmetry":
                x = batch["image"].to(args.device)
                labels = batch["label"].to(args.device)

                logits = model(x)
                preds = (torch.sigmoid(logits) > 0.5).float()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            elif args.task == "normals":
                x = batch["image"].to(args.device)
                gt_normals = batch["normals"].to(args.device)
                mask = batch["mask"].to(args.device)

                # Handle CLS token
                B, N, D = x.shape
                H_p = int((N - 1) ** 0.5)
                if (H_p * H_p) + 1 == N:
                    x = x[:, 1:, :]
                    N = N - 1
                else:
                    H_p = int(N**0.5)

                logits = model(x)

                # Reshape to spatial
                B, N, C = logits.shape
                logits_spatial = logits.permute(0, 2, 1).reshape(B, C, H_p, H_p)

                # Upsample to match GT
                logits_upsampled = torch.nn.functional.interpolate(
                    logits_spatial,
                    size=gt_normals.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

                # Compute Angular Error
                if mask.sum() > 0:
                    mae, rmse = compute_angular_error(
                        logits_upsampled, gt_normals, mask
                    )
                    total_angular_error += mae
                    total_rmse_error += rmse
                    num_batches += 1

    # Compute Metrics
    metrics = {
        "task": args.task,
        "backbone": args.backbone,
        "probe": args.probe,
        "layer": args.layer,
    }

    if args.task == "rotation":
        acc = accuracy_score(all_labels, all_preds)
        metrics["accuracy"] = acc
        print(f"Accuracy: {acc:.4f}")

    elif args.task == "symmetry":
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Exact Match Ratio (Subset Accuracy)
        subset_acc = accuracy_score(all_labels, all_preds)

        # F1 Scores
        f1_micro = f1_score(all_labels, all_preds, average="micro")
        f1_macro = f1_score(all_labels, all_preds, average="macro")

        metrics["subset_accuracy"] = subset_acc
        metrics["f1_micro"] = f1_micro
        metrics["f1_macro"] = f1_macro

        print(f"Subset Accuracy: {subset_acc:.4f}")
        print(f"F1 Micro: {f1_micro:.4f}")
        print(f"F1 Macro: {f1_macro:.4f}")

    elif args.task == "normals":
        avg_mae = total_angular_error / num_batches if num_batches > 0 else 0.0
        avg_rmse = total_rmse_error / num_batches if num_batches > 0 else 0.0
        metrics["mae_degrees"] = avg_mae
        metrics["rmse_degrees"] = avg_rmse
        print(f"Mean Angular Error (degrees): {avg_mae:.4f}")
        print(f"RMSE (degrees): {avg_rmse:.4f}")

    # Save Results
    output_path = Path(args.output_file)

    # Load existing results if file exists
    if output_path.exists():
        with open(output_path, "r") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = []
    else:
        results = []
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Append new result
    # Remove existing entry if duplicated key
    results = [
        r
        for r in results
        if not (
            r["task"] == args.task
            and r["backbone"] == args.backbone
            and r["probe"] == args.probe
            and r["layer"] == args.layer
        )
    ]
    results.append(metrics)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
