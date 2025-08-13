#!/bin/bash

# Safe Training Manager for Foundation-Sec-8B Fine-Tuning
# Provides easy stop/resume functionality

set -e

TRAINING_DATA=""
OUTPUT_DIR="./h100_output"
ACTION=""

show_help() {
    echo "Safe Training Manager"
    echo ""
    echo "Usage:"
    echo "  $0 start --data <path>              # Start new training"
    echo "  $0 resume                           # Auto-resume from latest checkpoint"
    echo "  $0 status                           # Check training status"
    echo "  $0 checkpoints                      # List available checkpoints"
    echo "  $0 stop                             # Stop running training safely"
    echo ""
    echo "Options:"
    echo "  --data <path>       Training data path"
    echo "  --output <path>     Output directory (default: ./h100_output")"
    echo ""
    echo "Examples:"
    echo "  $0 start --data ./my_data           # Start training"
    echo "  $0 stop                             # Stop training (Ctrl+C)"
    echo "  $0 resume                           # Resume from latest checkpoint"
    echo "  $0 status                           # Check if training is running"
}

check_training_status() {
    if pgrep -f "risk_fine_tuner_gradient_fixed.py" > /dev/null; then
        echo "[RUNNING] Training is currently RUNNING"
        echo "[INFO] Process ID: $(pgrep -f risk_fine_tuner_gradient_fixed.py)"
        echo "[STOP] To stop safely: Press Ctrl+C in the training terminal or run: $0 stop"
        return 0
    else
        echo "[STOPPED] Training is NOT running"
        return 1
    fi
}

stop_training() {
    if pgrep -f "risk_fine_tuner_gradient_fixed.py" > /dev/null; then
        echo "[STOP] Stopping training safely..."
        pkill -SIGINT -f "risk_fine_tuner_gradient_fixed.py"
        echo "[SUCCESS] SIGINT sent to training process"
        echo "[WAIT] Waiting for graceful shutdown..."
        sleep 3
        
        if pgrep -f "risk_fine_tuner_gradient_fixed.py" > /dev/null; then
            echo "[WAIT] Process still running, waiting more..."
            sleep 5
        fi
        
        if ! pgrep -f "risk_fine_tuner_gradient_fixed.py" > /dev/null; then
            echo "[SUCCESS] Training stopped successfully"
            list_checkpoints
        else
            echo "[ERROR] Training process still running. May need manual intervention."
        fi
    else
        echo "[ERROR] No training process found"
    fi
}

list_checkpoints() {
    echo "[CHECKPOINTS] Available checkpoints in $OUTPUT_DIR:"
    
    if [[ -d "$OUTPUT_DIR/checkpoints" ]]; then
        CHECKPOINTS=$(find "$OUTPUT_DIR/checkpoints" -name "checkpoint-*" -type d | sort -V)
        if [[ -n "$CHECKPOINTS" ]]; then
            echo "$CHECKPOINTS" | while read checkpoint; do
                STEP=$(basename "$checkpoint" | sed 's/checkpoint-//')
                SIZE=$(du -sh "$checkpoint" 2>/dev/null | cut -f1)
                echo "  [STEP] $STEP - $checkpoint ($SIZE)"
            done
            
            LATEST=$(echo "$CHECKPOINTS" | tail -1)
            echo ""
            echo "[RESUME] To resume from latest: $0 resume"
            echo "[LATEST] Latest checkpoint: $LATEST"
        else
            echo "  [NONE] No checkpoints found"
        fi
    else
        echo "  [NONE] No checkpoint directory found"
    fi
}

start_training() {
    if [[ -z "$TRAINING_DATA" ]]; then
        echo "[ERROR] Error: --data parameter required for start"
        echo "Example: $0 start --data ./my_training_data"
        exit 1
    fi
    
    if [[ ! -e "$TRAINING_DATA" ]]; then
        echo "[ERROR] Error: Training data path not found: $TRAINING_DATA"
        exit 1
    fi
    
    echo "[START] Starting training..."
    echo "[DATA] Data: $TRAINING_DATA"
    echo "[OUTPUT] Output: $OUTPUT_DIR"
    echo ""
    echo "[STOP] To stop safely: Press Ctrl+C or run '$0 stop' in another terminal"
    echo ""
    
    ./run_gradient_fixed.sh --data "$TRAINING_DATA" --output "$OUTPUT_DIR"
}

resume_training() {
    if [[ ! -d "$OUTPUT_DIR" ]]; then
        echo "[ERROR] Error: Output directory not found: $OUTPUT_DIR"
        echo "   No previous training found to resume"
        exit 1
    fi
    
    echo "[RESUME] Attempting to resume training..."
    echo "[OUTPUT] Output: $OUTPUT_DIR"
    
    # Check for training state
    if [[ -f "$OUTPUT_DIR/training_state.json" ]]; then
        echo "[INFO] Found training state file"
    fi
    
    list_checkpoints
    echo ""
    
    # Auto-detect training data from state file
    if [[ -f "$OUTPUT_DIR/train_fixed.jsonl" ]]; then
        echo "[RESUME] Resuming with auto-detected settings..."
        echo ""
        ./run_gradient_fixed.sh --data "$OUTPUT_DIR" --output "$OUTPUT_DIR"
    else
        echo "[ERROR] Error: Cannot auto-detect training data"
        echo "   Please use: $0 start --data <original_data_path>"
        exit 1
    fi
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        start)
            ACTION="start"
            shift
            ;;
        resume)
            ACTION="resume"
            shift
            ;;
        status)
            ACTION="status"
            shift
            ;;
        checkpoints)
            ACTION="checkpoints"
            shift
            ;;
        stop)
            ACTION="stop"
            shift
            ;;
        --data)
            TRAINING_DATA="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Execute action
case $ACTION in
    start)
        start_training
        ;;
    resume)
        resume_training
        ;;
    status)
        check_training_status
        ;;
    checkpoints)
        list_checkpoints
        ;;
    stop)
        stop_training
        ;;
    *)
        echo "[ERROR] No action specified"
        show_help
        exit 1
        ;;
esac 