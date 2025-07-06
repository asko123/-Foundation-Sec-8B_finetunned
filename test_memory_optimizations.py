#!/usr/bin/env python3
"""
Test Memory Optimizations - Verify that memory optimizations are working correctly
"""

import os
import torch
import gc
import psutil
from datetime import datetime

def test_pytorch_memory_config():
    """Test PyTorch memory configuration"""
    print("=== Testing PyTorch Memory Configuration ===")
    
    # Check if environment variable is set
    alloc_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'Not set')
    print(f"PYTORCH_CUDA_ALLOC_CONF: {alloc_conf}")
    
    if 'expandable_segments:True' in alloc_conf:
        print("[OK] PyTorch memory configuration is set correctly")
    else:
        print("[ERROR] PyTorch memory configuration needs to be set")
        return False
    
    return True

def test_cuda_availability():
    """Test CUDA availability and memory info"""
    print("\n=== Testing CUDA Availability ===")
    
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available")
        return False
    
    print("[OK] CUDA is available")
    
    # Get GPU info
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")
    
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
        print(f"GPU {i}: {gpu_name} - {gpu_memory:.2f}GB")
    
    return True

def test_memory_cleanup():
    """Test memory cleanup functions"""
    print("\n=== Testing Memory Cleanup ===")
    
    # Get initial memory usage
    if torch.cuda.is_available():
        initial_allocated = torch.cuda.memory_allocated() / (1024**3)
        initial_reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"Initial GPU memory - Allocated: {initial_allocated:.2f}GB, Reserved: {initial_reserved:.2f}GB")
    
    # Create some tensors to use memory
    print("Creating test tensors...")
    test_tensors = []
    try:
        for i in range(5):
            if torch.cuda.is_available():
                tensor = torch.randn(1000, 1000, device='cuda')
                test_tensors.append(tensor)
            else:
                tensor = torch.randn(1000, 1000)
                test_tensors.append(tensor)
    except Exception as e:
        print(f"Error creating tensors: {e}")
    
    # Check memory usage after creating tensors
    if torch.cuda.is_available():
        after_allocated = torch.cuda.memory_allocated() / (1024**3)
        after_reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"After creating tensors - Allocated: {after_allocated:.2f}GB, Reserved: {after_reserved:.2f}GB")
    
    # Clean up
    print("Cleaning up memory...")
    del test_tensors
    
    # Aggressive cleanup
    for _ in range(3):
        gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
    
    # Check memory usage after cleanup
    if torch.cuda.is_available():
        final_allocated = torch.cuda.memory_allocated() / (1024**3)
        final_reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"After cleanup - Allocated: {final_allocated:.2f}GB, Reserved: {final_reserved:.2f}GB")
        
        # Check if memory was freed
        if final_allocated <= initial_allocated + 0.1:  # Allow small tolerance
            print("[OK] Memory cleanup is working correctly")
            return True
        else:
            print("[ERROR] Memory cleanup may not be working properly")
            return False
    else:
        print("[OK] Memory cleanup completed (CPU mode)")
        return True

def test_system_memory():
    """Test system memory information"""
    print("\n=== Testing System Memory ===")
    
    # Get system memory info
    memory = psutil.virtual_memory()
    cpu_memory_total = memory.total / (1024**3)  # GB
    cpu_memory_used = memory.used / (1024**3)  # GB
    cpu_memory_percent = memory.percent
    
    print(f"System RAM - Total: {cpu_memory_total:.2f}GB, Used: {cpu_memory_used:.2f}GB ({cpu_memory_percent:.1f}%)")
    
    # Check if we have enough memory
    if cpu_memory_total >= 16:
        print("[OK] System has adequate RAM")
        return True
    else:
        print("[WARNING] System has limited RAM, consider using smaller batch sizes")
        return True

def main():
    """Run all memory optimization tests"""
    print(f"Memory Optimization Test - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    tests = [
        test_pytorch_memory_config,
        test_cuda_availability,
        test_system_memory,
        test_memory_cleanup
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"[ERROR] Test failed with error: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] All memory optimizations are working correctly!")
        print("[INFO] You can now run the optimized training script")
    else:
        print("[WARNING] Some tests failed. Please review the results above.")
        print("[TIP] Try running: export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 