#!/bin/bash

# Enhanced H100 Fine-Tuning Script with Checkpointing Support
# Optimized for H100 GPUs with time-limited training sessions (4-day limit)

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="training_output.log"
MEMORY_LOG_FILE="training_memory.log"

# Default values
TRAINING_DATA=""
OUTPUT_DIR="fine_tuning_output_h100"
RESUME_CHECKPOINT=""
NO_AUTO_RESUME=false

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Enhanced H100 Fine-Tuning Script with Checkpointing

Usage: $0 [OPTIONS]

OPTIONS:
    -d, --data PATH              Path to training data (required)
    -o, --output DIR             Output directory (default: fine_tuning_output_h100)
    -r, --resume PATH            Resume from specific checkpoint path
    -n, --no-auto-resume        Disable automatic checkpoint detection
    -h, --help                   Show this help message

EXAMPLES:
    # Start new training
    $0 --data ./training_data --output ./h100_output

    # Resume from specific checkpoint
    $0 --data ./training_data --output ./h100_output --resume ./h100_output/checkpoints/checkpoint-1000

    # Resume with auto-detection disabled
    $0 --data ./training_data --output ./h100_output --no-auto-resume

    # Quick start with defaults
    $0 --data ./training_data

FEATURES:
    ✓ H100-optimized batch sizes and checkpoint frequency
    ✓ Automatic checkpoint detection and resume
    ✓ Graceful handling of interruptions (Ctrl+C)
    ✓ Memory usage monitoring
    ✓ Progress tracking with time estimates
    ✓ Multiple checkpoint retention for safety

For 4-day time limits:
    - Training automatically saves every 25-200 steps (depending on GPU)
    - Time-based checkpointing every 30 minutes
    - Easy resume from any checkpoint
    - Handles interruptions gracefully

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--data)
            TRAINING_DATA="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -r|--resume)
            RESUME_CHECKPOINT="$2"
            shift 2
            ;;
        -n|--no-auto-resume)
            NO_AUTO_RESUME=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$TRAINING_DATA" ]]; then
    print_error "Training data path is required"
    show_usage
    exit 1
fi

if [[ ! -e "$TRAINING_DATA" ]]; then
    print_error "Training data path does not exist: $TRAINING_DATA"
    exit 1
fi

# Setup environment variables for H100 optimization
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export CUDA_LAUNCH_BLOCKING=0  # Disable for H100 performance
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export OMP_NUM_THREADS=8       # Optimize for H100

# Print configuration
print_status "H100 Fine-Tuning Configuration"
echo "=================================="
echo "Training Data: $TRAINING_DATA"
echo "Output Directory: $OUTPUT_DIR"
echo "Resume Checkpoint: ${RESUME_CHECKPOINT:-'Auto-detect'}"
echo "Auto-Resume: ${NO_AUTO_RESUME}"
echo "Log File: $LOG_FILE"
echo "Memory Log: $MEMORY_LOG_FILE"
echo "=================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to cleanup on exit
cleanup() {
    print_status "Cleaning up..."
    
    # Stop memory monitor if running
    if [[ -n "$MEMORY_PID" ]]; then
        print_status "Stopping memory monitor..."
        kill $MEMORY_PID 2>/dev/null || true
        wait $MEMORY_PID 2>/dev/null || true
        print_success "Memory monitor stopped"
    fi
    
    print_success "Cleanup completed"
}

# Setup trap for cleanup
trap cleanup EXIT INT TERM

# Clear GPU memory before starting
print_status "Clearing GPU memory..."
python3 -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print(f'GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
else:
    print('CUDA not available')
" 2>/dev/null || true

# Start memory monitoring in background
print_status "Starting memory monitor..."
python3 memory_monitor.py --interval 10 --log-file "$MEMORY_LOG_FILE" &
MEMORY_PID=$!
print_success "Memory monitor started (PID: $MEMORY_PID)"

# Build command line arguments for Python script
PYTHON_ARGS="--training-data \"$TRAINING_DATA\" --output \"$OUTPUT_DIR\""

if [[ -n "$RESUME_CHECKPOINT" ]]; then
    PYTHON_ARGS="$PYTHON_ARGS --resume-from-checkpoint \"$RESUME_CHECKPOINT\""
fi

if [[ "$NO_AUTO_RESUME" == "true" ]]; then
    PYTHON_ARGS="$PYTHON_ARGS --no-auto-resume"
fi

# Log start time
START_TIME=$(date)
print_status "Starting H100 training at: $START_TIME"

# Run the training with enhanced error handling
print_status "Launching H100-optimized training..."
echo "Command: python3 risk_fine_tuner_gradient_fixed.py $PYTHON_ARGS"
echo "========================================================"

if eval "python3 risk_fine_tuner_gradient_fixed.py $PYTHON_ARGS" 2>&1 | tee "$LOG_FILE"; then
    END_TIME=$(date)
    print_success "Training completed successfully!"
    print_status "Started: $START_TIME"
    print_status "Completed: $END_TIME"
    print_status "Logs saved to: $LOG_FILE"
    print_status "Memory usage: $MEMORY_LOG_FILE"
    
    # Show final checkpoint information
    if [[ -d "$OUTPUT_DIR/checkpoints" ]]; then
        LATEST_CHECKPOINT=$(find "$OUTPUT_DIR/checkpoints" -name "checkpoint-*" -type d | sort -V | tail -1)
        if [[ -n "$LATEST_CHECKPOINT" ]]; then
            CHECKPOINT_NUM=$(basename "$LATEST_CHECKPOINT" | cut -d'-' -f2)
            print_success "Final checkpoint: $LATEST_CHECKPOINT (step $CHECKPOINT_NUM)"
        fi
    fi
    
    if [[ -f "$OUTPUT_DIR/fixed_model.pkl" ]]; then
        print_success "Inference package: $OUTPUT_DIR/fixed_model.pkl"
    fi
    
else
    EXIT_CODE=$?
    END_TIME=$(date)
    
    if [[ $EXIT_CODE -eq 130 ]]; then
        print_warning "Training interrupted by user (Ctrl+C)"
        print_status "Checking for saved checkpoints..."
        
        if [[ -d "$OUTPUT_DIR/checkpoints" ]]; then
            CHECKPOINTS=$(find "$OUTPUT_DIR/checkpoints" -name "checkpoint-*" -type d | wc -l)
            if [[ $CHECKPOINTS -gt 0 ]]; then
                LATEST_CHECKPOINT=$(find "$OUTPUT_DIR/checkpoints" -name "checkpoint-*" -type d | sort -V | tail -1)
                print_success "Found $CHECKPOINTS checkpoint(s). Latest: $LATEST_CHECKPOINT"
                print_status "To resume training, run:"
                echo "  $0 --data \"$TRAINING_DATA\" --output \"$OUTPUT_DIR\" --resume \"$LATEST_CHECKPOINT\""
            fi
        fi
        
        if [[ -d "$OUTPUT_DIR/interrupted_checkpoint" ]]; then
            print_success "Interruption checkpoint saved: $OUTPUT_DIR/interrupted_checkpoint"
            print_status "To resume from interruption point, run:"
            echo "  $0 --data \"$TRAINING_DATA\" --output \"$OUTPUT_DIR\" --resume \"$OUTPUT_DIR/interrupted_checkpoint\""
        fi
    else
        print_error "Training failed with exit code: $EXIT_CODE"
        print_status "Check the log file for details: $LOG_FILE"
        
        # Check for error recovery checkpoint
        if [[ -d "$OUTPUT_DIR/error_checkpoint" ]]; then
            print_warning "Error recovery checkpoint saved: $OUTPUT_DIR/error_checkpoint"
            print_status "You may try to resume from this checkpoint if the error was temporary"
        fi
    fi
    
    print_status "Started: $START_TIME"
    print_status "Failed: $END_TIME"
    exit $EXIT_CODE
fi 