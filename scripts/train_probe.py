"""
scripts/train_probe.py
----------------------
Train probes on pre-computed features for GIQ tasks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
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
    parser.add_argument("--output_dir", type=str, default="experiments/probes")
    args = parser.parse_args()

    # Determine dimensions
    # Assume standard dimensions for now, or infer from first batch
    # CLIP/SigLIP: 768 (ViT-B), DINOv3: 768 (ViT-B)
    # But wait, CLIP projection dim might be 512.
    # We should infer input_dim dynamically.

    print(
        f"Loading {args.task} dataset with {args.backbone} features ({args.layer})..."
    )
    train_ds = get_dataset(args.task, args.backbone, args.layer, "train")
    val_ds = get_dataset(args.task, args.backbone, args.layer, "val")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Infer input dimension from a sample
    sample = train_ds[0]
    if args.task == "rotation":
        feat_dim = sample["img_a"].shape[-1]
    elif args.task == "symmetry":
        feat_dim = sample["image"].shape[-1]
    elif args.task == "normals":
        feat_dim = sample["image"].shape[-1]  # shape is [N, D] or [D]

    # Setup Model
    if args.task == "rotation":
        num_classes = 2
        loss_fn = nn.CrossEntropyLoss()
        if args.combine_method == "concat":
            input_dim = feat_dim * 2
        else:
            input_dim = feat_dim
    elif args.task == "symmetry":
        num_classes = 3
        loss_fn = nn.BCEWithLogitsLoss()
        input_dim = feat_dim
    elif args.task == "normals":
        num_classes = 3
        loss_fn = nn.L1Loss()  # Or CosineSimilarity
        input_dim = feat_dim

    print(f"Input dimension: {input_dim}")

    if args.probe == "linear":
        model = LinearProbe(input_dim, num_classes)
    elif args.probe == "mlp":
        model = MLPProbe(input_dim, hidden_dim=input_dim, num_classes=num_classes)
    elif args.probe == "dense":
        if args.task != "normals":
            raise ValueError("Dense probe only for normals task")
        model = DenseProbe(input_dim, num_classes)

    model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training Loop
    best_val_loss = float("inf")
    output_path = Path(args.output_dir) / f"{args.task}_{args.backbone}_{args.probe}"
    output_path.mkdir(parents=True, exist_ok=True)

    criterion = loss_fn  # Alias for clarity

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
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

            elif args.task == "symmetry":
                x = batch["image"].to(args.device)
                labels = batch["label"].to(args.device)

            elif args.task == "normals":
                x = batch["image"].to(args.device)  # Patch tokens [B, N, D]
                # Ground truth normals [B, 3, H, W]
                gt_normals = batch["normals"].to(args.device)
                mask = batch["mask"].to(args.device)

                # Handle CLS token if present (e.g., N=197 for ViT-B/16)
                B, N, D = x.shape
                H_p = int((N - 1) ** 0.5)
                if (H_p * H_p) + 1 == N:
                    x = x[:, 1:, :]  # Drop CLS token
                    N = N - 1
                else:
                    H_p = int(N**0.5)
                    if H_p * H_p != N:
                        # Fallback or error? Maybe reshape is not square?
                        # Just proceed and let reshape fail if so.
                        pass

            optimizer.zero_grad()
            logits = model(x)

            loss = 0
            if args.task == "normals":
                # Dense prediction logic
                # Logits: [B, N, 3] -> Permute to [B, 3, N] -> Reshape/Interpolate
                B, N, C = logits.shape
                # Reshape to spatial
                logits_spatial = logits.permute(0, 2, 1).reshape(B, C, H_p, H_p)

                # Upsample to match GT [B, 3, 224, 224]
                logits_upsampled = torch.nn.functional.interpolate(
                    logits_spatial,
                    size=gt_normals.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

                # Mask out background if available
                if mask.sum() > 0:
                    loss = criterion(logits_upsampled[mask], gt_normals[mask])
                else:
                    loss = torch.tensor(0.0, device=args.device, requires_grad=True)

            else:
                loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Simple accuracy logging
            if args.task == "rotation":
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            elif args.task == "symmetry":
                # Multi-label accuracy? Or exact match?
                # Use strict threshold 0.5 for now
                preds = (torch.sigmoid(logits) > 0.5).float()
                # Exact match
                correct += (preds == labels).all(dim=1).sum().item()
                total += labels.size(0)

            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        acc = correct / total if total > 0 else 0.0
        print(f"Train Loss: {avg_loss:.4f}, Train Acc: {acc:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
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
                elif args.task == "symmetry":
                    x = batch["image"].to(args.device)
                    labels = batch["label"].to(args.device)
                elif args.task == "normals":
                    x = batch["image"].to(args.device)
                    gt_normals = batch["normals"].to(args.device)
                    mask = batch["mask"].to(args.device)

                    # Handle CLS token
                    B, N, D = x.shape
                    H_p = int((N - 1) ** 0.5)
                    if (H_p * H_p) + 1 == N:
                        x = x[:, 1:, :]  # Drop CLS token
                        N = N - 1
                    else:
                        H_p = int(N**0.5)

                logits = model(x)

                if args.task == "normals":
                    B, N, C = logits.shape
                    logits_spatial = logits.permute(0, 2, 1).reshape(B, C, H_p, H_p)
                    logits_upsampled = torch.nn.functional.interpolate(
                        logits_spatial,
                        size=gt_normals.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    if mask.sum() > 0:
                        loss = criterion(logits_upsampled[mask], gt_normals[mask])
                    else:
                        loss = torch.tensor(0.0, device=args.device)
                else:
                    loss = criterion(logits, labels)

                val_loss += loss.item()

                if args.task == "rotation":
                    preds = torch.argmax(logits, dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
                elif args.task == "symmetry":
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    val_correct += (preds == labels).all(dim=1).sum().item()
                    val_total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), output_path / "best_model.pth")
            print("Saved best model.")


if __name__ == "__main__":
    main()
