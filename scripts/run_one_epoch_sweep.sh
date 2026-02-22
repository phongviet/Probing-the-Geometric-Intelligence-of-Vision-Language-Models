#!/bin/bash
# scripts/run_one_epoch_sweep.sh
# Run a one-epoch sweep over all training configurations.

# Ensure we are in the project root
cd "$(dirname "$0")/.."

BACKBONES=("clip" "siglip2" "dinov3")
LAYERS=("global" "local")
PROBES=("linear" "mlp")
ROTATION_COMBINE=("concat" "diff" "mult")

echo "Starting one-epoch sweep for backbones: ${BACKBONES[*]}"

for BACKBONE in "${BACKBONES[@]}"; do
    echo "========================================================"
    echo "Running one-epoch sweep for backbone: $BACKBONE"
    echo "========================================================"

    for LAYER in "${LAYERS[@]}"; do
        # 1. Mental Rotation (Linear + MLP, all combine methods)
        for PROBE in "${PROBES[@]}"; do
            for COMBINE in "${ROTATION_COMBINE[@]}"; do
                OUTPUT_DIR="experiments/probes_rotation_${COMBINE}"
                echo "[$BACKBONE][$LAYER] Rotation ($PROBE, $COMBINE) -> $OUTPUT_DIR..."
                conda run -n geoprobe python scripts/train_probe.py \
                    --task rotation \
                    --backbone "$BACKBONE" \
                    --probe "$PROBE" \
                    --layer "$LAYER" \
                    --combine_method "$COMBINE" \
                    --device cuda \
                    --epochs 1 \
                    --output_dir "$OUTPUT_DIR"
            done
        done

        # 2. Symmetry Detection (Linear + MLP)
        for PROBE in "${PROBES[@]}"; do
            OUTPUT_DIR="experiments/probes_symmetry_${LAYER}"
            echo "[$BACKBONE][$LAYER] Symmetry ($PROBE) -> $OUTPUT_DIR..."
            conda run -n geoprobe python scripts/train_probe.py \
                --task symmetry \
                --backbone "$BACKBONE" \
                --probe "$PROBE" \
                --layer "$LAYER" \
                --device cuda \
                --epochs 1 \
                --output_dir "$OUTPUT_DIR"
        done

        # 3. Surface Normals (Dense Probe)
        if [ "$LAYER" = "local" ]; then
            OUTPUT_DIR="experiments/probes_normals_${LAYER}"
            echo "[$BACKBONE][$LAYER] Normals (dense) -> $OUTPUT_DIR..."
            conda run -n geoprobe python scripts/train_probe.py \
                --task normals \
                --backbone "$BACKBONE" \
                --probe dense \
                --layer "$LAYER" \
                --device cuda \
                --epochs 1 \
                --output_dir "$OUTPUT_DIR"
        else
            echo "[$BACKBONE][$LAYER] Normals skipped (requires local features)."
        fi
    done
done

echo "One-epoch sweep complete."
