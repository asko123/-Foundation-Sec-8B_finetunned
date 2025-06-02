#!/bin/bash

# Fix libstdc++ library path
export LD_LIBRARY_PATH="/sw/external/conda-4.8.5/env:/sw/external/conda-4.8.5/lib:$LD_LIBRARY_PATH"

# Fix threading issues that cause segmentation faults
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export NPY_NUM_BUILD_JOBS=1
export TORCH_NUM_THREADS=1

# Run Python with all fixes applied
python "$@" 