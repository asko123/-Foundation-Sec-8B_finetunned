#!/bin/bash

# Run the gradient-fixed version of the fine-tuner
echo "=== GRADIENT-FIXED FINE-TUNER ==="

# Set environment variables for better stability
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1

# Parse arguments
TRAINING_DATA=""
OUTPUT_DIR="fine_tuning_output_fixed"

while [[ $# -gt 0 ]]; do
    case $1 in
        --training-data)
            TRAINING_DATA="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 --training-data <path> [--output <dir>]"
            echo ""
            echo "Options:"
            echo "  --training-data <path>  Path to training data (required)"
            echo "  --output <dir>          Output directory (default: fine_tuning_output_fixed)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$TRAINING_DATA" ]; then
    echo "Error: --training-data is required"
    exit 1
fi

if [ ! -e "$TRAINING_DATA" ]; then
    echo "Error: Training data not found at $TRAINING_DATA"
    exit 1
fi

echo "Training data: $TRAINING_DATA"
echo "Output directory: $OUTPUT_DIR"

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"

# Run the fixed fine-tuner
python risk_fine_tuner_gradient_fixed.py \
    --training-data "$TRAINING_DATA" \
    --output "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed!"
    exit 1
fi 