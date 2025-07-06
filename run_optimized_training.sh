#!/bin/bash

# Optimized Training Script with Memory Management
# This script runs the risk fine-tuner with memory optimizations and monitoring

echo "=== OPTIMIZED RISK & PII FINE-TUNER ==="
echo "Setting up memory-optimized training environment..."

# Set PyTorch memory allocation configuration
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_LAUNCH_BLOCKING=1

# Additional PyTorch optimizations
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,expandable_segments:True"

# Set memory-efficient Python settings
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

echo "Environment variables set:"
echo "- PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo "- CUDA_LAUNCH_BLOCKING: $CUDA_LAUNCH_BLOCKING"

# Function to check if training data exists
check_training_data() {
    if [ ! -e "$1" ]; then
        echo "[ERROR] Training data not found at $1"
        echo "Please provide a valid path to your training data"
        exit 1
    fi
}

# Function to start memory monitoring
start_memory_monitor() {
    echo "[MONITOR] Starting memory monitor..."
    python memory_monitor.py --interval 10 --log-file "training_memory.log" &
    MONITOR_PID=$!
    echo "Memory monitor started with PID: $MONITOR_PID"
}

# Function to stop memory monitoring
stop_memory_monitor() {
    if [ ! -z "$MONITOR_PID" ]; then
        echo "[STOP] Stopping memory monitor..."
        kill $MONITOR_PID 2>/dev/null
        echo "Memory monitor stopped"
    fi
}

# Function to cleanup on exit
cleanup() {
    echo "[CLEANUP] Cleaning up..."
    stop_memory_monitor
    
    # Clear CUDA cache
    python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"
    
    echo "Cleanup completed"
}

# Set up cleanup on script exit
trap cleanup EXIT

# Parse command line arguments
TRAINING_DATA=""
OUTPUT_DIR="fine_tuning_output"
MONITOR_MEMORY=true

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
        --no-monitor)
            MONITOR_MEMORY=false
            shift
            ;;
        --help)
            echo "Usage: $0 --training-data <path> [--output <dir>] [--no-monitor]"
            echo ""
            echo "Options:"
            echo "  --training-data <path>  Path to training data (required)"
            echo "  --output <dir>          Output directory (default: fine_tuning_output)"
            echo "  --no-monitor           Disable memory monitoring"
            echo "  --help                 Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$TRAINING_DATA" ]; then
    echo "[ERROR] --training-data is required"
    echo "Use --help for usage information"
    exit 1
fi

# Check if training data exists
check_training_data "$TRAINING_DATA"

# Start memory monitoring if enabled
if [ "$MONITOR_MEMORY" = true ]; then
    start_memory_monitor
fi

# Clear any existing GPU memory
echo "[CONFIG] Clearing GPU memory..."
python -c "import torch; torch.cuda.empty_cache(); torch.cuda.ipc_collect() if torch.cuda.is_available() else None"

# Run the training with optimizations
echo "[INFO] Starting optimized training..."
echo "Training data: $TRAINING_DATA"
echo "Output directory: $OUTPUT_DIR"

# Run the actual training
python risk_fine_tuner.py \
    --training-data "$TRAINING_DATA" \
    --output "$OUTPUT_DIR" \
    2>&1 | tee "training_output.log"

# Check training result
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Training completed successfully!"
    echo "[FILE] Model saved to: $OUTPUT_DIR"
    echo "[LOG] Training log: training_output.log"
    if [ "$MONITOR_MEMORY" = true ]; then
        echo "[MEMORY] Memory usage log: training_memory.log"
    fi
else
    echo "[ERROR] Training failed!"
    echo "[LOG] Check training_output.log for details"
    if [ "$MONITOR_MEMORY" = true ]; then
        echo "[MEMORY] Memory usage log: training_memory.log"
    fi
    exit 1
fi

echo "[DONE] Training process completed!" 