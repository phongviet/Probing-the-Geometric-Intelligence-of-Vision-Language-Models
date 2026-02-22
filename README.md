# Probing the Geometric Intelligence of Vision-Language Models

It provides a comprehensive probing framework designed to evaluate the extent to which modern visual foundation models—such as **SigLIP 2** and **DINOv3**—implicitly encode 3D spatial relationships, depth, and non-Euclidean geometry within their frozen latent spaces.

The codebase is built around the **Geometric Intelligence Quotient (GIQ)** benchmark and the **Probe3D** evaluation protocol.

---

## Features

- **Decoupled Extraction & Probing Pipeline:** Designed to run efficiently on consumer-grade hardware (e.g., an NVIDIA GTX 1660 Ti with 6GB VRAM) by separating massive backbone feature extraction from lightweight probe training.
- **Visual Foundation Models Support:** Pre-integrated with CLIP, SigLIP 2, and DINOv3 architectures.
- **Multiple Probe Architectures:** Easily toggle between Linear Probes (for disentanglement evaluation), Non-Linear Probes (MLPs), and Dense Probes ($1 \times 1$ convolutions) for spatial/pixel-wise tasks.
- **GIQ Benchmark Tasks:** 
  - **Single-View Surface Normal Estimation:** Reconstruct 2.5D surface properties from frozen dense patch tokens.
  - **Mental Rotation:** Determine if two highly-rotated polyhedra represent the same underlying 3D object (chirality checks).
  - **Symmetry Detection:** Multi-label classification evaluating point-group geometric intelligence (e.g., 4-fold or 5-fold rotation).

---

## Environment & Setup

The project uses `conda` to manage its Python 3.11 environment and strictly pins dependencies to ensure exact reproducibility.

```bash
# Clone the repository
git clone https://github.com/phongviet/Probing-the-Geometric-Intelligence-of-Vision-Language-Models.git
cd Probing-the-Geometric-Intelligence-of-Vision-Language-Models

# Create and activate the conda environment
conda env create -f environment.yml
conda activate geoprobe
```

*Note: Ensure you have manually downloaded the 90GB of rendered images from the GIQ benchmark into `data/giq/renderings/`. You can download the 3D meshes using `conda run -n geoprobe python scripts/download_giq_meshes.py`.*

---

## Usage Pipeline

The repository operates in three distinct phases:

### 1. Feature Extraction (Hardware-Aware)
Run standard ViT encoders over the renderings and offload `[CLS]` (global) and spatial patch (local) tokens to disk as compressed `.npz` arrays. This guarantees VRAM limits are respected.

```bash
# Example: Extract test-split features for SigLIP 2
conda run -n geoprobe python scripts/extract_features.py --model siglip2 --split test
```
*Tip: To batch-run all extractions at once, use `bash scripts/run_all_extractions.sh`.*

### 2. Train Lightweight Probes
Train standard logistic regression (linear), MLP, or dense probes on top of the frozen representations for the GIQ tasks.

```bash
# Example: Train a linear probe on the Mental Rotation task using local CLIP tokens
conda run -n geoprobe python scripts/train_probe.py \
    --task rotation \
    --backbone clip \
    --probe linear \
    --layer local \
    --epochs 10 \
    --batch_size 32
```
*Tip: You can automate the entire training sweep across all models, probes, and tasks by executing `bash scripts/run_experiments.sh`.*

### 3. Evaluate and Visualize Results
Evaluate your trained probes and generate a JSON file containing comprehensive evaluation metrics (RMSE, Exact Subset Accuracy, and F1 scores).

```bash
# Example: Evaluate the results and save to experiments/results.json
conda run -n geoprobe python scripts/evaluate.py --task rotation --backbone siglip2 --probe linear --layer local
```

*You can run all evaluations concurrently via `bash scripts/run_evaluation.sh`.*

---

## Repository Layout

```text
├── src/                   # Core Python package
│   ├── data/              # Dataset PyTorch classes (MentalRotation, Symmetry, Normals, and FeatureMixins)
│   └── models/            # Vision Encoder Featurizers (CLIP, SigLIP2, DINOv3) and Probing architectures
├── scripts/               # Standalone execution scripts
│   ├── test_dataloaders.py # Smoke test dataset logic (no 90GB dataset required)
│   ├── extract_features.py # ViT latent space extraction
│   ├── train_probe.py      # Probe training loop
│   └── evaluate.py         # Test-set evaluation and metrics
├── data/                  # Root directory for GIQ meshes, normals, renderings, and splits
├── experiments/           # Saved probe models, weights, and JSON metrics
├── giq-benchmark/         # Upstream JSON definitions (read-only)
├── notebooks/             # Kaggle/Colab feature extraction notebooks
├── Report.md              # Detailed final ICLR methodology, results, and discussion
└── AGENTS.md              # Internal development standards, strict typing, and linting rules
```

---

## Testing and Verification

There is no formal `pytest` suite. Instead, use the pre-built standalone smoke-testing script which intelligently mocks data on-the-fly to test dataset loaders.

```bash
# Run ALL dataset smoke tests
conda run -n geoprobe python scripts/test_dataloaders.py

# Run standard code linting and formatting
conda run -n geoprobe ruff check --fix src/ scripts/
conda run -n geoprobe ruff format src/ scripts/
```