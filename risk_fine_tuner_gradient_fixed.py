#!/usr/bin/env python3
"""
Gradient-Fixed Risk & PII Fine-Tuner - A tool for fine-tuning the Foundation-Sec-8B model.
COMPLETELY FIXED VERSION - Addresses the "element 0 of tensor does not require grad" error.

This version includes comprehensive gradient handling fixes for LoRA training.
"""

# Set PyTorch memory allocation configuration before importing torch
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Standard library imports
import json
import csv
import random
import pickle
import gc
import traceback
import re
import argparse
import sys
from typing import List, Dict, Any, Optional, Tuple, Union

# Third-party imports
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import numpy as np

# Local imports
from risk_fine_tuner_enhanced import (
    scan_folder_for_files,
    extract_text_from_excel,
    extract_text_from_csv,
    analyze_content_for_risk_category,
    analyze_content_for_pii,
    process_raw_data_to_training_examples,
    process_folder_for_training_data
)
from risk_fine_tuner_constants import (
    L2,
    MACRO_RISKS,
    PII_PROTECTION_CATEGORIES,
    PII_TYPES,
    PRIVACY_CLASSIFICATIONS,
    SENSITIVITY_LEVELS
)

def format_all_categories_for_prompt() -> str:
    """Format all categories (risk and PII) for inclusion in prompts."""
    categories_text = "PART 1: SECURITY RISK CATEGORIES\n\n"
    categories_text += "L2 Categories:\n"
    for key, value in L2.items():
        categories_text += f"{key}. {value}\n"
    
    categories_text += "\nMacro Risks for each L2 Category:\n"
    for key, themes in MACRO_RISKS.items():
        categories_text += f"\n{key}. {L2[key]}:\n"
        for theme in themes:
            categories_text += f"   - {theme}\n"
    
    categories_text += "\n\nPART 2: PII PROTECTION CATEGORIES\n\n"
    categories_text += "PII Protection Categories:\n"
    for key, value in PII_PROTECTION_CATEGORIES.items():
        categories_text += f"{key}: {value}\n"
    
    categories_text += "\nCommon PII Types:\n"
    for pii_type in PII_TYPES:
        categories_text += f"- {pii_type}\n"
            
    return categories_text

def setup_model_for_training(model_name: str, device: str, use_quantization: bool = False) -> Tuple[Any, Any]:
    """
    Set up model and tokenizer for training with proper gradient handling.
    COMPLETELY FIXED VERSION for gradient requirements.
    """
    print(f"[SETUP] Loading model: {model_name}")
    
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # FIXED: Simplified model loading without aggressive optimizations
    model_kwargs = {
        "trust_remote_code": True,
        "use_cache": False,
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
    }
    
    # Only add device_map for multi-GPU setups
    if device == "cuda" and torch.cuda.device_count() > 1:
        model_kwargs["device_map"] = "auto"
    
    # FIXED: Disable quantization to avoid gradient issues
    if use_quantization and device == "cuda":
        # Get VRAM to determine quantization type
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        if vram_gb < 6:  # Very low VRAM - use 4bit
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_quant_storage=torch.uint8
            )
            model_kwargs["quantization_config"] = quantization_config
            print(f"[SETUP] Using 4-bit quantization for {vram_gb:.1f}GB VRAM")
        else:
            # For >6GB VRAM, use full precision to avoid gradient issues
            print(f"[SETUP] Using full precision for {vram_gb:.1f}GB VRAM")
    else:
        print("[SETUP] Using full precision (no quantization)")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    # FIXED: Proper model preparation
    if use_quantization and device == "cuda" and hasattr(model, 'config') and getattr(model.config, 'quantization_config', None):
        print("[SETUP] Preparing quantized model for training...")
        model = prepare_model_for_kbit_training(model)
    
    # Ensure model is in training mode
    model.train()
    
    # Move to device if not using device_map
    if device == "cuda" and "device_map" not in model_kwargs:
        model = model.to(device)
    
    print(f"[SETUP] Model loaded on device: {device}")
    return model, tokenizer

def setup_lora_config(vram_gb: float) -> LoraConfig:
    """Create LoRA configuration based on available VRAM."""
    if vram_gb < 6:  # Very low VRAM
        r = 4
        alpha = 8
        target_modules = ["q_proj", "v_proj"]
        print(f"[LORA] Low VRAM config: r={r}, alpha={alpha}, modules={len(target_modules)}")
    elif vram_gb < 12:  # Medium VRAM
        r = 8
        alpha = 16
        target_modules = ["q_proj", "v_proj", "k_proj"]
        print(f"[LORA] Medium VRAM config: r={r}, alpha={alpha}, modules={len(target_modules)}")
    else:  # High VRAM
        r = 16
        alpha = 32
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        print(f"[LORA] High VRAM config: r={r}, alpha={alpha}, modules={len(target_modules)}")
    
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        # FIXED: Ensure proper initialization
        init_lora_weights=True,
        use_rslora=False,  # Disable RSLoRA to avoid gradient issues
        use_dora=False,    # Disable DoRA to avoid gradient issues
    )

def apply_lora_and_verify_gradients(model, lora_config) -> Any:
    """Apply LoRA and verify gradients are properly set up."""
    print("[LORA] Applying LoRA configuration...")
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # FIXED: Comprehensive gradient setup
    print("[LORA] Setting up gradients...")
    
    # First, disable gradients for all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Then, enable gradients only for LoRA parameters
    for name, param in model.named_parameters():
        if any(lora_key in name for lora_key in ['lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B']):
            param.requires_grad = True
            print(f"[LORA] Enabled gradients for: {name}")
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Verify we have trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"[LORA] Total parameters: {total_params:,}")
    print(f"[LORA] Trainable parameters: {trainable_params:,}")
    print(f"[LORA] Trainable percentage: {100 * trainable_params / total_params:.4f}%")
    
    if trainable_params == 0:
        raise RuntimeError("No trainable parameters found after LoRA application!")
    
    return model

def format_chat_example(example, tokenizer):
    """Format chat examples for training."""
    try:
        if isinstance(example["messages"], list):
            # Use tokenizer's chat template if available
            if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
                formatted_text = tokenizer.apply_chat_template(
                    example["messages"], 
                    tokenize=False, 
                    add_generation_prompt=False
                )
            else:
                # Fallback to manual formatting
                formatted_text = ""
                for msg in example["messages"]:
                    if msg["role"] == "system":
                        formatted_text += f"System: {msg['content']}\n"
                    elif msg["role"] == "user":
                        formatted_text += f"User: {msg['content']}\n"
                    elif msg["role"] == "assistant":
                        formatted_text += f"Assistant: {msg['content']}\n"
            
            # Tokenize with proper settings
            tokens = tokenizer(
                formatted_text,
                truncation=True,
                max_length=512,  # Reasonable max length
                padding=False,   # Don't pad during tokenization
                return_tensors=None  # Return lists, not tensors
            )
            
            return {"input_ids": tokens["input_ids"]}
        else:
            return {"input_ids": []}
    except Exception as e:
        print(f"[ERROR] Failed to format example: {e}")
        return {"input_ids": []}

class CustomDataCollator:
    """Custom data collator that properly handles gradients."""
    
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, examples):
        """Collate examples into batches."""
        # Extract input_ids from examples
        input_ids = [example["input_ids"] for example in examples if example["input_ids"]]
        
        if not input_ids:
            # Return empty batch if no valid examples
            return {
                "input_ids": torch.tensor([[self.tokenizer.pad_token_id]]),
                "labels": torch.tensor([[self.tokenizer.pad_token_id]]),
                "attention_mask": torch.tensor([[1]])
            }
        
        # Pad sequences
        max_len = min(max(len(ids) for ids in input_ids), self.max_length)
        
        batch_input_ids = []
        batch_attention_mask = []
        
        for ids in input_ids:
            # Truncate if too long
            if len(ids) > max_len:
                ids = ids[:max_len]
            
            # Pad if too short
            attention_mask = [1] * len(ids)
            while len(ids) < max_len:
                ids.append(self.tokenizer.pad_token_id)
                attention_mask.append(0)
            
            batch_input_ids.append(ids)
            batch_attention_mask.append(attention_mask)
        
        # Convert to tensors
        batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
        batch_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long)
        
        # For causal LM, labels are the same as input_ids
        labels = batch_input_ids.clone()
        
        # Set padding tokens to -100 so they're ignored in loss computation
        labels[batch_attention_mask == 0] = -100
        
        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": labels
        }

def fine_tune_model_fixed(training_data_path: str, output_dir: str = "fine_tuning_output") -> Optional[str]:
    """
    Fine-tune model with comprehensive gradient fixes.
    COMPLETELY FIXED VERSION for gradient requirements.
    """
    print("=== GRADIENT-FIXED FINE-TUNING ===")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare training data (using existing functions)
    from risk_fine_tuner import (
        detect_data_format, 
        load_training_file, 
        process_folder_for_training_data,
        format_risk_example,
        format_pii_example,
        process_batch
    )
    
    # Detect and load training data
    data_format = detect_data_format(training_data_path)
    print(f"[DATA] Detected format: {data_format}")
    
    training_examples = []
    if data_format == 'raw_folder':
        temp_training_file = process_folder_for_training_data(training_data_path, os.path.join(output_dir, "extracted_data"))
        training_examples = load_training_file(temp_training_file)
    elif data_format in ['raw_file', 'formatted_file']:
        training_examples = load_training_file(training_data_path)
    elif os.path.isdir(training_data_path):
        for filename in os.listdir(training_data_path):
            file_path = os.path.join(training_data_path, filename)
            if os.path.isfile(file_path):
                training_examples.extend(load_training_file(file_path))
    
    if not training_examples:
        raise ValueError("No training examples found!")
    
    print(f"[DATA] Found {len(training_examples)} training examples")
    
    # Split data
    random.shuffle(training_examples)
    split_idx = int(len(training_examples) * 0.9)
    train_examples = training_examples[:split_idx]
    eval_examples = training_examples[split_idx:]
    
    print(f"[DATA] Training: {len(train_examples)}, Eval: {len(eval_examples)}")
    
    # Create training files
    train_file = os.path.join(output_dir, "train_fixed.jsonl")
    eval_file = os.path.join(output_dir, "eval_fixed.jsonl")
    
    categories = format_all_categories_for_prompt()
    
    print("[DATA] Creating training files...")
    with open(train_file, 'w', encoding='utf-8') as f:
        process_batch(train_examples, f)
    
    with open(eval_file, 'w', encoding='utf-8') as f:
        process_batch(eval_examples, f)
    
    # Setup device and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if device == "cuda" else 0
    
    print(f"[DEVICE] Using {device}, VRAM: {vram_gb:.1f}GB")
    
    # FIXED: Only use quantization for very low VRAM
    use_quantization = device == "cuda" and vram_gb < 6
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_for_training(
        "fdtn-ai/Foundation-Sec-8B", 
        device, 
        use_quantization=use_quantization
    )
    
    # Setup LoRA
    lora_config = setup_lora_config(vram_gb)
    model = apply_lora_and_verify_gradients(model, lora_config)
    
    # Load and process datasets
    print("[DATA] Loading datasets...")
    train_dataset = load_dataset('json', data_files=train_file, split='train')
    eval_dataset = load_dataset('json', data_files=eval_file, split='train')
    
    # Format datasets
    def format_function(example):
        return format_chat_example(example, tokenizer)
    
    train_dataset = train_dataset.map(format_function, remove_columns=["messages"])
    eval_dataset = eval_dataset.map(format_function, remove_columns=["messages"])
    
    # Filter out empty examples
    train_dataset = train_dataset.filter(lambda x: len(x["input_ids"]) > 0)
    eval_dataset = eval_dataset.filter(lambda x: len(x["input_ids"]) > 0)
    
    print(f"[DATA] Processed datasets - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    # Training configuration
    batch_size = 1 if vram_gb < 8 else 2
    gradient_accumulation_steps = 32 if vram_gb < 8 else 16
    
    print(f"[TRAIN] Batch size: {batch_size}, Gradient accumulation: {gradient_accumulation_steps}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=2,  # Reduced epochs to avoid overfitting
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=3e-4,  # Higher learning rate for LoRA
        weight_decay=0.01,
        warmup_ratio=0.03,
        logging_steps=10,
        eval_steps=50,
        save_steps=100,
        save_strategy="steps",
        eval_strategy="steps",
        # FIXED: Precision settings
        fp16=device == "cuda" and not use_quantization,
        bf16=False,
        # FIXED: Memory and stability
        gradient_checkpointing=False,  # Disable to avoid gradient issues
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        max_grad_norm=1.0,
        # Logging
        logging_first_step=True,
        report_to="none",
        # Hardware
        no_cuda=device == "cpu",
        optim="adamw_torch",
        seed=42,
        # FIXED: Disable problematic settings
        prediction_loss_only=True,
        skip_memory_metrics=True,
        load_best_model_at_end=False,  # Disable to avoid issues
        save_total_limit=2,
    )
    
    # Custom data collator
    data_collator = CustomDataCollator(tokenizer, max_length=512)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Test forward pass before training
    print("[TEST] Testing forward pass...")
    try:
        test_batch = data_collator([train_dataset[0]])
        if device == "cuda":
            test_batch = {k: v.cuda() for k, v in test_batch.items()}
        
        model.eval()
        with torch.no_grad():
            outputs = model(**test_batch)
        
        model.train()
        print("[TEST] ✓ Forward pass successful")
    except Exception as e:
        print(f"[TEST] ✗ Forward pass failed: {e}")
        raise
    
    # Start training
    print("[TRAIN] Starting training...")
    trainer.train()
    
    # Save model
    print("[SAVE] Saving model...")
    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    # Create inference package
    pickle_path = os.path.join(output_dir, "fixed_model.pkl")
    
    from risk_fine_tuner import create_inference_package
    inference_package = create_inference_package(final_model_path)
    
    with open(pickle_path, 'wb') as f:
        pickle.dump(inference_package, f)
    
    print(f"[SUCCESS] Training completed!")
    print(f"[FILE] Model: {final_model_path}")
    print(f"[FILE] Inference package: {pickle_path}")
    
    return pickle_path

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Gradient-Fixed Risk & PII Fine-Tuner")
    parser.add_argument("--training-data", required=True, help="Path to training data")
    parser.add_argument("--output", default="fine_tuning_output_fixed", help="Output directory")
    
    args = parser.parse_args()
    
    try:
        result = fine_tune_model_fixed(args.training_data, args.output)
        
        if result:
            print(f"\n✅ SUCCESS! Model saved to: {result}")
            return 0
        else:
            print(f"\n❌ FAILED!")
            return 1
            
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 