#!/bin/bash
set -e

# Run extraction for all models and splits
# Usage: ./scripts/run_all_extractions.sh

MODELS=("clip" "siglip2" "dinov3")
SPLITS=("train" "val" "test")
BATCH_SIZE=32

for model in "${MODELS[@]}"; do
    for split in "${SPLITS[@]}"; do
        echo "Running extraction for Model: $model, Split: $split"
        conda run -n geoprobe python scripts/extract_features.py \
            --model "$model" \
            --split "$split" \
            --batch_size "$BATCH_SIZE" \
            --fp16
            # Add --limit 10 for testing if needed
    done
done

echo "All extractions complete."

# Run validation
echo "Running validation..."
for model in "${MODELS[@]}"; do
    # Map model to directory name (heuristic based on default names)
    if [ "$model" == "clip" ]; then dir="openai__clip-vit-base-patch16"; fi
    if [ "$model" == "siglip2" ]; then dir="google__siglip2-base-patch16-224"; fi
    if [ "$model" == "dinov3" ]; then dir="facebook__dinov3-base"; fi
    
    for split in "${SPLITS[@]}"; do
        echo "Validating $model $split..."
        conda run -n geoprobe python scripts/validate_features.py \
            --output_dir "data/giq/features/$dir/$split" || echo "Validation failed or no files for $model $split"
    done
done
