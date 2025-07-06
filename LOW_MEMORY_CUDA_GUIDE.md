# Low Memory CUDA Fine-Tuning Guide

This guide explains how to use the optimized fine-tuning system for low-memory CUDA environments.

## 🔧 **System Requirements**

### Minimum Requirements
- **GPU**: NVIDIA GPU with CUDA support (6GB+ VRAM recommended)
- **RAM**: 16GB+ system RAM
- **Storage**: 50GB+ free space
- **Python**: 3.8+

### GPU Memory Categories
| VRAM | Category | Optimizations Applied |
|------|----------|----------------------|
| <6GB | Very Low Memory | 4-bit quantization, batch=1, grad_accum=64+ |
| 6-8GB | Low Memory | 4-bit quantization, batch=1, grad_accum=32-64 |
| 8-12GB | Mid-Low Memory | 8-bit quantization, batch=1, grad_accum=16-32 |
| 12-16GB | Standard Memory | 8-bit quantization, batch=1, grad_accum=16 |
| 16GB+ | High Memory | Standard optimizations |

## 📦 **Installation**

```bash
# Install required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate peft bitsandbytes pandas openpyxl psutil

# Verify installation
python test_low_memory_cuda.py
```

## 🚀 **Quick Start**

### 1. Test Your System
```bash
# Test memory optimizations
python test_memory_optimizations.py

# Test low-memory CUDA specific features
python test_low_memory_cuda.py
```

### 2. Run Optimized Training
```bash
# Using the optimized script (recommended)
./run_optimized_training.sh --training-data /path/to/your/data

# Or run directly
python risk_fine_tuner.py --training-data /path/to/your/data --output ./output
```

## ⚙️ **Automatic Optimizations**

The system automatically applies optimizations based on your GPU memory:

### Ultra-Low Memory (<8GB)
- **Quantization**: 4-bit with NF4
- **LoRA**: r=2, alpha=4, 1 module (q_proj only)
- **Batch Size**: 1
- **Gradient Accumulation**: 64 steps
- **Epochs**: 2 (reduced)
- **Checkpoints**: 1 (minimal)
- **CPU Offloading**: Enabled
- **Memory Cleanup**: Every 5 steps

### Low Memory (8-12GB)
- **Quantization**: 8-bit
- **LoRA**: r=4, alpha=8, 2 modules
- **Batch Size**: 1
- **Gradient Accumulation**: 32-48 steps
- **Epochs**: 2
- **Checkpoints**: 2
- **Memory Cleanup**: Every 10 steps

### Standard Memory (12-16GB)
- **Quantization**: 8-bit
- **LoRA**: r=8, alpha=16, 3 modules
- **Batch Size**: 1
- **Gradient Accumulation**: 16-32 steps
- **Epochs**: 3
- **Checkpoints**: 3
- **Memory Cleanup**: Every 20 steps

## 🛠 **Manual Configuration**

You can override automatic settings by modifying the code:

```python
# In risk_fine_tuner.py, modify the setup_device() function
def setup_device():
    # Force specific settings
    device = "cuda"
    bs = 1  # Batch size
    ga_steps = 64  # Gradient accumulation steps
    return device, bs, ga_steps
```

## 📊 **Memory Monitoring**

### Real-time Monitoring
The system includes built-in memory monitoring:
- GPU memory usage tracking
- Automatic cleanup when memory usage > 85%
- Memory summaries at each training stage

### Log Files
- `training_output.log` - Training progress
- `training_memory.log` - Memory usage over time

## 🔍 **Troubleshooting**

### Common Issues

#### 1. CUDA Out of Memory
```bash
[ERROR] CUDA out of memory. Tried to allocate XYZ MiB
```
**Solutions:**
- Run `python test_low_memory_cuda.py` to verify setup
- Ensure no other processes are using GPU memory
- Try restarting Python/clearing GPU memory

#### 2. Slow Training
**Causes:**
- High gradient accumulation (trade-off for memory)
- CPU offloading enabled

**Normal behavior for low-memory setups**

#### 3. Model Quality Issues
**Solutions:**
- Increase gradient accumulation steps
- Use higher LoRA rank if memory allows
- Train for more epochs

### Emergency Recovery
If training fails with OOM:
1. System automatically switches to ultra-conservative settings
2. Disables evaluation and checkpointing
3. Increases gradient accumulation to 64+ steps
4. Retries training

## 📈 **Performance Expectations**

### Training Speed
| GPU Memory | Relative Speed | Notes |
|------------|----------------|-------|
| <8GB | 0.2-0.4x | Slow due to high grad accumulation |
| 8-12GB | 0.4-0.6x | Moderate speed |
| 12-16GB | 0.6-0.8x | Good speed |
| 16GB+ | 0.8-1.0x | Near-optimal speed |

### Memory Usage
- **Base Model**: ~7-15GB (depending on quantization)
- **LoRA Parameters**: 50-200MB
- **Training Overhead**: 2-4GB
- **Total**: Usually fits in 8GB+ with optimizations

## 🔧 **Advanced Configuration**

### Environment Variables
```bash
# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Additional optimizations
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1
```

### Custom LoRA Settings
```python
# Ultra-minimal for <6GB
lora_config = LoraConfig(
    r=1,
    lora_alpha=2,
    target_modules=["q_proj"]
)

# Aggressive for 16GB+
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj"]
)
```

## 📋 **Best Practices**

### 1. Data Preparation
- Process data before training to avoid memory overhead
- Use smaller training datasets for initial testing
- Monitor memory during data loading

### 2. Training Strategy
- Start with minimal settings and gradually increase
- Use evaluation sparingly on low-memory systems
- Save checkpoints less frequently

### 3. System Management
- Close other applications during training
- Monitor system temperature
- Use adequate cooling for extended training

## 🚨 **Limitations**

### What Works Well
- ✅ Risk categorization training
- ✅ PII detection training
- ✅ Small to medium datasets
- ✅ LoRA fine-tuning

### What May Struggle
- ❌ Very large datasets (>100K examples)
- ❌ Full model fine-tuning
- ❌ Multiple simultaneous training runs
- ❌ Complex multi-GPU setups

## 📞 **Support**

If you encounter issues:
1. Run the test scripts to diagnose problems
2. Check the memory logs for bottlenecks
3. Try reducing batch size or increasing gradient accumulation
4. Consider using a smaller model or dataset for testing

## 🔄 **Updates**

The low-memory optimizations are continuously improved. Key features:
- Automatic GPU detection and optimization
- Progressive memory cleanup
- Emergency recovery mechanisms
- Comprehensive monitoring and logging 