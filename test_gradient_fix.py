#!/usr/bin/env python3
"""
Test script to verify gradient requirements are properly set for LoRA training.
This helps debug the "element 0 of tensor does not require grad" error.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def test_gradient_setup():
    """Test that gradients are properly set up for LoRA training."""
    print("=== TESTING GRADIENT SETUP ===")
    
    # Check CUDA
    if torch.cuda.is_available():
        device = "cuda"
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Using CUDA with {vram_gb:.1f}GB VRAM")
        
        # Set quantization based on VRAM
        if vram_gb < 8:
            use_4bit = True
            use_8bit = False
        elif vram_gb < 12:
            use_4bit = False
            use_8bit = True
        else:
            use_4bit = False
            use_8bit = False
            
        print(f"Quantization: 4bit={use_4bit}, 8bit={use_8bit}")
    else:
        device = "cpu"
        use_4bit = False
        use_8bit = False
        print("Using CPU")
    
    # Load model with quantization if needed
    model_kwargs = {
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        "device_map": "auto" if device == "cuda" else None,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        "use_cache": False,
    }
    
    if device == "cuda" and (use_4bit or use_8bit):
        if use_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["quantization_config"] = quantization_config
            print("Using 4-bit quantization")
        elif use_8bit:
            model_kwargs["load_in_8bit"] = True
            print("Using 8-bit quantization")
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "fdtn-ai/Foundation-Sec-8B",
        **model_kwargs
    )
    
    print("Model loaded successfully")
    
    # Prepare for quantized training if needed
    if device == "cuda" and (use_4bit or use_8bit):
        print("Preparing model for quantized training...")
        model = prepare_model_for_kbit_training(model)
        model.train()
        
        # Set gradients for adapter parameters
        for name, param in model.named_parameters():
            if "lora" in name.lower() or "adapter" in name.lower():
                param.requires_grad = True
                
        print("Model prepared for quantized training")
    else:
        model.train()
        print("Model set to training mode")
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )
    
    print("Applying LoRA...")
    model = get_peft_model(model, lora_config)
    
    # Ensure LoRA parameters require gradients
    for name, param in model.named_parameters():
        if "lora" in name.lower() or "adapter" in name.lower():
            param.requires_grad = True
    
    # Check trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.4f}%")
    
    if trainable_params == 0:
        print("❌ ERROR: No trainable parameters found!")
        return False
    else:
        print("✅ SUCCESS: Found trainable parameters")
    
    # Test gradient computation
    print("\nTesting gradient computation...")
    tokenizer = AutoTokenizer.from_pretrained("fdtn-ai/Foundation-Sec-8B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create a simple test input
    test_text = "This is a test input for gradient computation."
    inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
    
    if device == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    try:
        # Forward pass
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        
        print(f"Forward pass successful, loss: {loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        
        # Check if gradients were computed
        grad_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_count += 1
        
        print(f"Gradients computed for {grad_count} parameters")
        
        if grad_count > 0:
            print("✅ SUCCESS: Gradients computed successfully!")
            return True
        else:
            print("❌ ERROR: No gradients computed!")
            return False
            
    except Exception as e:
        print(f"❌ ERROR during gradient computation: {e}")
        return False

if __name__ == "__main__":
    success = test_gradient_setup()
    if success:
        print("\n🎉 Gradient setup test PASSED! You can now run the full training.")
    else:
        print("\n💥 Gradient setup test FAILED! Please check the configuration.") 