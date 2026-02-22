#!/bin/bash
# scripts/run_evaluation.sh
# Evaluate all trained probes and aggregate results

# Ensure we are in the project root
cd "$(dirname "$0")/.."

BACKBONES=("clip" "siglip2" "dinov3")
LAYERS=("global" "local")
PROBES=("linear" "mlp")
ROTATION_COMBINE=("concat" "diff" "mult")
OUTPUT_FILE="experiments/results.json"

echo "Starting evaluation..."

for BACKBONE in "${BACKBONES[@]}"; do
    echo "========================================================"
    echo "Evaluating backbone: $BACKBONE"
    echo "========================================================"

    for LAYER in "${LAYERS[@]}"; do
        # 1. Mental Rotation (Linear + MLP, all combine methods)
        for PROBE in "${PROBES[@]}"; do
            for COMBINE in "${ROTATION_COMBINE[@]}"; do
                MODEL_DIR="experiments/probes_rotation_${COMBINE}"
                echo "[$BACKBONE][$LAYER] Rotation ($PROBE, $COMBINE) -> $MODEL_DIR..."
                conda run -n geoprobe python scripts/evaluate.py \
                    --task rotation \
                    --backbone "$BACKBONE" \
                    --probe "$PROBE" \
                    --layer "$LAYER" \
                    --combine_method "$COMBINE" \
                    --model_dir "$MODEL_DIR" \
                    --device cuda \
                    --output_file "$OUTPUT_FILE"
            done
        done

        # 2. Symmetry Detection (Linear + MLP)
        for PROBE in "${PROBES[@]}"; do
            MODEL_DIR="experiments/probes_symmetry_${LAYER}"
            echo "[$BACKBONE][$LAYER] Symmetry ($PROBE) -> $MODEL_DIR..."
            conda run -n geoprobe python scripts/evaluate.py \
                --task symmetry \
                --backbone "$BACKBONE" \
                --probe "$PROBE" \
                --layer "$LAYER" \
                --model_dir "$MODEL_DIR" \
                --device cuda \
                --output_file "$OUTPUT_FILE"
        done

        # 3. Surface Normals (Dense only)
        if [ "$LAYER" = "local" ]; then
            MODEL_DIR="experiments/probes_normals_${LAYER}"
            echo "[$BACKBONE][$LAYER] Normals (dense) -> $MODEL_DIR..."
            conda run -n geoprobe python scripts/evaluate.py \
                --task normals \
                --backbone "$BACKBONE" \
                --probe dense \
                --layer "$LAYER" \
                --model_dir "$MODEL_DIR" \
                --device cuda \
                --output_file "$OUTPUT_FILE"
        else
            echo "[$BACKBONE][$LAYER] Normals skipped (requires local features)."
        fi
    done
done

echo "Evaluation complete. Results saved to $OUTPUT_FILE"
