#!/bin/bash
# scripts/run_experiments.sh
# Automate training runs for GIQ Probes

# Ensure we are in the project root
cd "$(dirname "$0")/.."

echo "Starting experiments..."

# 1. Mental Rotation
echo "Training Mental Rotation (Linear Probe)..."
conda run -n geoprobe python scripts/train_probe.py --task rotation --backbone clip --probe linear --epochs 10 --combine_method concat

echo "Training Mental Rotation (MLP Probe)..."
conda run -n geoprobe python scripts/train_probe.py --task rotation --backbone clip --probe mlp --epochs 10 --combine_method concat

# 2. Symmetry Detection
echo "Training Symmetry Detection (Linear Probe)..."
conda run -n geoprobe python scripts/train_probe.py --task symmetry --backbone clip --probe linear --epochs 10

# 3. Surface Normals (Dense Probe)
# Requires 'local' features. If not available, this might fail or warn.
echo "Training Surface Normals (Dense Probe)..."
conda run -n geoprobe python scripts/train_probe.py --task normals --backbone clip --probe dense --layer local --epochs 10

echo "Experiments complete."
