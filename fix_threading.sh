#!/bin/bash

echo "Setting up environment with threading fixes..."

# Fix the libstdc++ issue
export LD_LIBRARY_PATH="/sw/external/conda-4.8.5/env:/sw/external/conda-4.8.5/lib:$LD_LIBRARY_PATH"

# Fix OpenBLAS threading issues
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Additional threading controls
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

# Prevent numpy from using multiple threads
export NPY_NUM_BUILD_JOBS=1

# PyTorch specific threading
export TORCH_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "Environment variables set:"
echo "  LD_LIBRARY_PATH includes conda libraries"
echo "  All threading libraries limited to 1 thread"
echo "  OpenBLAS threading disabled"

# Test Python
echo ""
echo "Testing Python with threading fixes..."
python -c "import numpy as np; print('NumPy OK:', np.__version__)"

echo ""
echo "Environment ready! Run your scripts with:"
echo "  python risk_fine_tuner.py"
echo "  python risk_inference.py" 