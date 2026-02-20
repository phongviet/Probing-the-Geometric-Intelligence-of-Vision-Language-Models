# GIQ Experiments

Experimental repository exploring ideas from the paper **"Probing the Geometric Intelligence of Vision-Language Models"** (ICLR 2026). This project implements probes for geometric tasks like mental rotation and symmetry detection.

## Setup

```bash
conda env create -f environment.yml
conda activate geoprobe
```

## Usage

**1. Extract Features**
Extract features from a model (e.g., CLIP, SigLIP, DINOv3).
```bash
python scripts/extract_features.py --model clip --split test
```

**2. Train Probes**
Run linear probes on the extracted features.
```bash
bash scripts/run_experiments.sh
```

**3. Notebooks**
See `notebooks/kaggle_feature_extraction.ipynb` for running feature extraction on Kaggle.

## Structure
- `src/models`: Featurizer implementations.
- `src/data`: Dataset classes.
- `scripts`: Utility scripts for extraction and evaluation.
