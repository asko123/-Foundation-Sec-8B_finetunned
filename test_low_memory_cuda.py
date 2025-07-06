#!/usr/bin/env python3
"""
Low Memory CUDA Test - Comprehensive testing for low-memory CUDA environments
"""

import os
import torch
import gc
import psutil
import json
from datetime import datetime
from typing import Dict, Any

# Set memory optimizations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def aggressive_cleanup():
    """Perform aggressive memory cleanup."""
    for _ in range(5):
        gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()

def get_memory_info() -> Dict[str, float]:
    """Get comprehensive memory information."""
    info = {}
    
    # System memory
    memory = psutil.virtual_memory()
    info['system_total_gb'] = memory.total / (1024**3)
    info['system_used_gb'] = memory.used / (1024**3)
    info['system_percent'] = memory.percent
    
    # GPU memory
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_total_gb'] = props.total_memory / (1024**3)
        info['gpu_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
        info['gpu_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
        info['gpu_free_gb'] = info['gpu_total_gb'] - info['gpu_allocated_gb']
    else:
        info['gpu_name'] = "No CUDA GPU"
        info['gpu_total_gb'] = 0
        info['gpu_allocated_gb'] = 0
        info['gpu_reserved_gb'] = 0
        info['gpu_free_gb'] = 0
    
    return info

def test_memory_allocation_patterns():
    """Test different memory allocation patterns for low-memory environments."""
    print("\n=== Testing Memory Allocation Patterns ===")
    
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available - cannot test GPU memory patterns")
        return False
    
    initial_info = get_memory_info()
    print(f"[INFO] Initial GPU memory: {initial_info['gpu_allocated_gb']:.2f}GB allocated")
    
    try:
        # Test small tensor allocations (typical for low-memory environments)
        print("[TEST] Creating small test tensors...")
        tensors = []
        
        for i in range(10):
            # Small tensors that should work on low-memory GPUs
            tensor = torch.randn(100, 100, device='cuda', dtype=torch.float16)
            tensors.append(tensor)
        
        mid_info = get_memory_info()
        print(f"[INFO] After allocation: {mid_info['gpu_allocated_gb']:.2f}GB allocated")
        
        # Cleanup test
        del tensors
        aggressive_cleanup()
        
        final_info = get_memory_info()
        print(f"[INFO] After cleanup: {final_info['gpu_allocated_gb']:.2f}GB allocated")
        
        # Check if cleanup was effective
        if final_info['gpu_allocated_gb'] <= initial_info['gpu_allocated_gb'] + 0.1:
            print("[OK] Memory allocation/cleanup test passed")
            return True
        else:
            print("[WARNING] Memory cleanup may not be fully effective")
            return False
            
    except torch.cuda.OutOfMemoryError as e:
        print(f"[ERROR] OOM during basic allocation test: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Allocation test failed: {e}")
        return False

def test_quantization_support():
    """Test quantization support for memory reduction."""
    print("\n=== Testing Quantization Support ===")
    
    if not torch.cuda.is_available():
        print("[INFO] Skipping quantization test - no CUDA GPU")
        return True
    
    try:
        # Test if bitsandbytes is available for quantization
        try:
            import bitsandbytes as bnb
            print("[OK] BitsAndBytes available for quantization")
            quantization_available = True
        except ImportError:
            print("[WARNING] BitsAndBytes not available - install with: pip install bitsandbytes")
            quantization_available = False
        
        # Test basic FP16 support
        try:
            test_tensor = torch.randn(100, 100, device='cuda', dtype=torch.float16)
            result = test_tensor @ test_tensor.T
            del test_tensor, result
            aggressive_cleanup()
            print("[OK] FP16 operations supported")
            fp16_available = True
        except Exception as e:
            print(f"[ERROR] FP16 not supported: {e}")
            fp16_available = False
        
        return quantization_available and fp16_available
        
    except Exception as e:
        print(f"[ERROR] Quantization test failed: {e}")
        return False

def test_low_memory_model_loading():
    """Test loading a small model with low-memory optimizations."""
    print("\n=== Testing Low-Memory Model Loading ===")
    
    if not torch.cuda.is_available():
        print("[INFO] Skipping model loading test - no CUDA GPU")
        return True
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Use a small model for testing
        model_name = "microsoft/DialoGPT-small"  # Small model for testing
        
        print(f"[TEST] Loading small model: {model_name}")
        initial_info = get_memory_info()
        
        # Load with memory optimizations
        model_kwargs = {
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
            "use_cache": False,
        }
        
        # Add quantization if available
        vram_gb = initial_info['gpu_total_gb']
        if vram_gb < 8:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model_kwargs["quantization_config"] = quantization_config
                print("[CONFIG] Using 4-bit quantization for <8GB GPU")
            except ImportError:
                model_kwargs["load_in_8bit"] = True
                print("[CONFIG] Using 8-bit quantization (BitsAndBytes not available)")
        elif vram_gb < 12:
            model_kwargs["load_in_8bit"] = True
            print("[CONFIG] Using 8-bit quantization for low-memory GPU")
        
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        loaded_info = get_memory_info()
        memory_used = loaded_info['gpu_allocated_gb'] - initial_info['gpu_allocated_gb']
        print(f"[INFO] Model loaded, used {memory_used:.2f}GB GPU memory")
        
        # Test a simple forward pass
        test_input = tokenizer("Hello, this is a test.", return_tensors="pt")
        if torch.cuda.is_available():
            test_input = {k: v.cuda() for k, v in test_input.items()}
        
        with torch.no_grad():
            outputs = model(**test_input)
        
        print("[OK] Model loading and forward pass successful")
        
        # Cleanup
        del model, tokenizer, outputs, test_input
        aggressive_cleanup()
        
        final_info = get_memory_info()
        print(f"[INFO] After cleanup: {final_info['gpu_allocated_gb']:.2f}GB allocated")
        
        return True
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"[ERROR] OOM during model loading test: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Model loading test failed: {e}")
        return False

def test_gradient_accumulation():
    """Test gradient accumulation for memory efficiency."""
    print("\n=== Testing Gradient Accumulation ===")
    
    if not torch.cuda.is_available():
        print("[INFO] Skipping gradient accumulation test - no CUDA GPU")
        return True
    
    try:
        # Create a simple model for testing
        model = torch.nn.Linear(100, 10).cuda().half()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = torch.nn.MSELoss()
        
        # Test with gradient accumulation
        accumulation_steps = 4
        batch_size = 8
        
        print(f"[TEST] Testing gradient accumulation: {accumulation_steps} steps, batch size {batch_size}")
        
        model.train()
        optimizer.zero_grad()
        
        total_loss = 0
        for step in range(accumulation_steps):
            # Create small batch
            x = torch.randn(batch_size, 100, device='cuda', dtype=torch.float16)
            y = torch.randn(batch_size, 10, device='cuda', dtype=torch.float16)
            
            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, y) / accumulation_steps  # Scale loss
            
            # Backward pass
            loss.backward()
            total_loss += loss.item()
            
            # Memory cleanup for low-memory environments
            del x, y, outputs
            
        # Optimizer step after accumulation
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"[OK] Gradient accumulation successful, average loss: {total_loss:.4f}")
        
        # Cleanup
        del model, optimizer, criterion
        aggressive_cleanup()
        
        return True
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"[ERROR] OOM during gradient accumulation test: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Gradient accumulation test failed: {e}")
        return False

def generate_memory_report():
    """Generate a comprehensive memory report."""
    print("\n=== Memory Environment Report ===")
    
    info = get_memory_info()
    
    # System information
    print(f"System RAM: {info['system_total_gb']:.1f}GB total, {info['system_used_gb']:.1f}GB used ({info['system_percent']:.1f}%)")
    
    # GPU information
    if torch.cuda.is_available():
        print(f"GPU: {info['gpu_name']}")
        print(f"GPU Memory: {info['gpu_total_gb']:.1f}GB total, {info['gpu_allocated_gb']:.2f}GB allocated")
        print(f"GPU Free: {info['gpu_free_gb']:.2f}GB available")
        
        # Memory category
        vram_gb = info['gpu_total_gb']
        if vram_gb < 6:
            category = "Very Low Memory"
            recommendations = [
                "Use 4-bit quantization",
                "Batch size = 1, gradient accumulation = 64+",
                "Disable evaluation during training",
                "Use CPU offloading"
            ]
        elif vram_gb < 8:
            category = "Low Memory"
            recommendations = [
                "Use 4-bit quantization",
                "Batch size = 1, gradient accumulation = 32-64",
                "Minimal LoRA parameters (r=2)"
            ]
        elif vram_gb < 12:
            category = "Mid-Low Memory"
            recommendations = [
                "Use 8-bit quantization",
                "Batch size = 1, gradient accumulation = 16-32",
                "Conservative LoRA parameters (r=4)"
            ]
        elif vram_gb < 16:
            category = "Standard Memory"
            recommendations = [
                "Use 8-bit quantization",
                "Batch size = 1, gradient accumulation = 16",
                "Standard LoRA parameters (r=8)"
            ]
        else:
            category = "High Memory"
            recommendations = [
                "Can use larger LoRA parameters",
                "Standard training configurations should work"
            ]
        
        print(f"\nMemory Category: {category}")
        print("Recommended optimizations:")
        for rec in recommendations:
            print(f"  - {rec}")
    else:
        print("GPU: No CUDA GPU available")
        print("Recommendation: Use CPU training with very small models")

def main():
    """Run comprehensive low-memory CUDA tests."""
    print("=" * 60)
    print("LOW MEMORY CUDA ENVIRONMENT TEST")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Generate initial report
    generate_memory_report()
    
    # Run tests
    tests = [
        ("Memory Allocation Patterns", test_memory_allocation_patterns),
        ("Quantization Support", test_quantization_support),
        ("Low-Memory Model Loading", test_low_memory_model_loading),
        ("Gradient Accumulation", test_gradient_accumulation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            if test_func():
                passed += 1
                print(f"[PASS] {test_name}")
            else:
                print(f"[FAIL] {test_name}")
        except Exception as e:
            print(f"[ERROR] {test_name} failed with exception: {e}")
    
    # Final report
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] All low-memory CUDA tests passed!")
        print("[INFO] Your system is ready for low-memory fine-tuning")
    else:
        print("[WARNING] Some tests failed. Your system may need optimization.")
        print("[TIP] Install missing packages: pip install bitsandbytes transformers")
    
    # Final memory state
    print("\nFinal Memory State:")
    final_info = get_memory_info()
    if torch.cuda.is_available():
        print(f"GPU Memory: {final_info['gpu_allocated_gb']:.2f}GB allocated")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 