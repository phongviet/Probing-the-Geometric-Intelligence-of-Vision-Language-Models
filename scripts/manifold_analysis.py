"""
scripts/manifold_analysis.py
----------------------------
Analyze the manifold structure of feature representations for rotating shapes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.feature_loader import FeatureLoader


def compute_smoothness(features):
    """
    Compute trajectory smoothness using cosine similarity between adjacent views.
    features: [N, D] numpy array
    Returns: average cosine similarity (higher is smoother)
    """
    # Normalized features
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    feats_norm = features / (norms + 1e-8)

    # Dot product between adjacent
    # sim[i] = dot(f[i], f[i+1])
    sims = np.sum(feats_norm[:-1] * feats_norm[1:], axis=1)

    return np.mean(sims)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone", type=str, required=True, choices=["clip", "siglip2", "dinov3"]
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--shape_id",
        type=str,
        default=None,
        help="Specific shape ID to analyze. If None, picks random.",
    )
    parser.add_argument(
        "--layer", type=str, default="global", choices=["global"]
    )  # Manifold analysis typically on global feats
    parser.add_argument("--output_dir", type=str, default="experiments/manifolds")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading features for {args.backbone} ({args.split})...")
    loader = FeatureLoader(backbone=args.backbone, split=args.split, layer=args.layer)

    # Find a shape
    if args.shape_id is None:
        # List available files in loader root
        # This is a bit hacky as loader doesn't expose list of shapes directly
        # But we know the structure: root/*.npz
        available_files = list(loader.root.glob("*.npz"))
        if not available_files:
            print(f"No features found in {loader.root}")
            return
        # Pick one
        import random

        selected_file = random.choice(available_files)
        shape_id = selected_file.stem
    else:
        shape_id = args.shape_id

    print(f"Analyzing shape: {shape_id}")

    if not loader.has_features(shape_id):
        print(f"Features not found for {shape_id}")
        return

    # Load all views
    # Use internal cache mechanism to avoid repeated disk reads
    loader._load_to_cache(shape_id)
    features = loader._cache[shape_id]  # [N_views, D]

    print(f"Loaded {features.shape[0]} views with dimension {features.shape[1]}")

    # Compute Smoothness
    smoothness = compute_smoothness(features)
    print(f"Trajectory Smoothness (Avg Cos Sim): {smoothness:.4f}")

    # Projections
    print("Computing PCA...")
    pca = PCA(n_components=2)
    pca_proj = pca.fit_transform(features)

    print("Computing t-SNE...")
    # Perplexity should be less than N_views
    perp = min(30, features.shape[0] - 1)
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    tsne_proj = tsne.fit_transform(features)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Color by view index to show trajectory
    indices = np.arange(features.shape[0])

    # PCA
    sc1 = axes[0].scatter(
        pca_proj[:, 0], pca_proj[:, 1], c=indices, cmap="viridis", s=50, alpha=0.8
    )
    axes[0].set_title(f"PCA Projection\nSmoothness: {smoothness:.3f}")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    plt.colorbar(sc1, ax=axes[0], label="View Index")

    # t-SNE
    sc2 = axes[1].scatter(
        tsne_proj[:, 0], tsne_proj[:, 1], c=indices, cmap="viridis", s=50, alpha=0.8
    )
    axes[1].set_title(f"t-SNE Projection\n{args.backbone} - {shape_id}")
    axes[1].set_xlabel("Dim 1")
    axes[1].set_ylabel("Dim 2")
    plt.colorbar(sc2, ax=axes[1], label="View Index")

    plt.tight_layout()
    plot_path = output_dir / f"manifold_{args.backbone}_{shape_id}.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
