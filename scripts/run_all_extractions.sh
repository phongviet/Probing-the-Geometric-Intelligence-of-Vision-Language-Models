#!/bin/bash
set -e

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root directory (parent of script directory)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to the project root directory
cd "$PROJECT_ROOT"

echo "Running from project root: $PROJECT_ROOT"

# Run extraction for all models and splits

MODELS=("clip" "siglip2" "dinov3")
SPLITS=("train" "val" "test")
BATCH_SIZE=32

for model in "${MODELS[@]}"; do
    for split in "${SPLITS[@]}"; do
        echo "Running extraction for Model: $model, Split: $split"
        # Using || true to prevent script from stopping if one model fails (e.g. auth error)
        conda run -n geoprobe python scripts/extract_features.py \
            --model "$model" \
            --split "$split" \
            --batch_size "$BATCH_SIZE" \
            --fp16 || echo "Extraction failed for $model $split"
            # Add --limit 10 for testing if needed
    done
done

echo "All extractions complete."

# Run validation
echo "Running validation..."
for model in "${MODELS[@]}"; do
    dir=""
    # Map model to directory name (heuristic based on default names)
    if [ "$model" == "clip" ]; then dir="openai__clip-vit-base-patch16"; fi
    if [ "$model" == "siglip2" ]; then dir="google__siglip2-base-patch16-224"; fi
    if [ "$model" == "dinov3" ]; then dir="facebook__dinov3-vitb16-pretrain-lvd1689m"; fi
    
    if [ -z "$dir" ]; then
        echo "Unknown model directory for $model"
        continue
    fi

    for split in "${SPLITS[@]}"; do

        echo "Validating $model $split..."
        conda run -n geoprobe python scripts/validate_features.py \
            --output_dir "data/giq/features/$dir/$split" || echo "Validation failed or no files for $model $split"
    done
done
