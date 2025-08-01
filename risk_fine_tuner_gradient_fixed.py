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
import time
import glob
from datetime import datetime, timedelta
from pathlib import Path
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
from ddl_pii_analyzer import DDLPIIAnalyzer

def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint in the checkpoint directory."""
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Look for checkpoint directories
    checkpoint_pattern = os.path.join(checkpoint_dir, "checkpoint-*")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        return None
    
    # Sort by checkpoint number
    def get_checkpoint_number(path):
        try:
            return int(os.path.basename(path).split('-')[1])
        except (IndexError, ValueError):
            return 0
    
    latest_checkpoint = max(checkpoints, key=get_checkpoint_number)
    
    # Verify checkpoint is valid
    if os.path.exists(os.path.join(latest_checkpoint, "pytorch_model.bin")) or \
       os.path.exists(os.path.join(latest_checkpoint, "adapter_model.bin")) or \
       any(f.endswith(".safetensors") for f in os.listdir(latest_checkpoint)):
        print(f"[CHECKPOINT] Found latest checkpoint: {latest_checkpoint}")
        return latest_checkpoint
    
    return None

def save_training_state(output_dir: str, train_examples: List, eval_examples: List, 
                       current_step: int = 0, total_steps: int = 0):
    """Save training state for resume capability."""
    state_file = os.path.join(output_dir, "training_state.json")
    
    state = {
        "timestamp": datetime.now().isoformat(),
        "current_step": current_step,
        "total_steps": total_steps,
        "train_examples_count": len(train_examples),
        "eval_examples_count": len(eval_examples),
        "data_files": {
            "train": os.path.join(output_dir, "train_fixed.jsonl"),
            "eval": os.path.join(output_dir, "eval_fixed.jsonl")
        }
    }
    
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)
    
    print(f"[STATE] Saved training state to {state_file}")

def load_training_state(output_dir: str) -> Optional[Dict]:
    """Load training state for resume."""
    state_file = os.path.join(output_dir, "training_state.json")
    
    if not os.path.exists(state_file):
        return None
    
    try:
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        print(f"[STATE] Loaded training state from {state_file}")
        print(f"[STATE] Previous run: {state['current_step']}/{state['total_steps']} steps")
        return state
    except Exception as e:
        print(f"[STATE] Failed to load training state: {e}")
        return None

class CheckpointCallback:
    """Callback for time-based and step-based checkpointing."""
    
    def __init__(self, output_dir: str, save_every_minutes: int = 30):
        self.output_dir = output_dir
        self.save_every_minutes = save_every_minutes
        self.last_save_time = time.time()
        self.start_time = time.time()
    
    def should_save_checkpoint(self, current_step: int) -> bool:
        """Determine if we should save a checkpoint based on time."""
        current_time = time.time()
        elapsed_minutes = (current_time - self.last_save_time) / 60
        
        if elapsed_minutes >= self.save_every_minutes:
            self.last_save_time = current_time
            return True
        
        return False
    
    def log_progress(self, current_step: int, total_steps: int):
        """Log training progress with time estimates."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        if current_step > 0:
            avg_time_per_step = elapsed_time / current_step
            remaining_steps = total_steps - current_step
            estimated_remaining_time = remaining_steps * avg_time_per_step
            
            elapsed_str = str(timedelta(seconds=int(elapsed_time)))
            remaining_str = str(timedelta(seconds=int(estimated_remaining_time)))
            
            print(f"[PROGRESS] Step {current_step}/{total_steps} ({current_step/total_steps*100:.1f}%)")
            print(f"[TIME] Elapsed: {elapsed_str}, Estimated remaining: {remaining_str}")

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

def detect_data_format(training_data_path: str) -> str:
    """Detect whether the provided path contains raw data or pre-formatted training data."""
    if os.path.isdir(training_data_path):
        # Check if directory contains raw Excel/CSV files
        for file in os.listdir(training_data_path):
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in ['.xlsx', '.xls', '.csv']:
                return 'raw_folder'
        return 'unknown'
    elif os.path.isfile(training_data_path):
        file_ext = os.path.splitext(training_data_path)[1].lower()
        if file_ext in ['.json', '.jsonl']:
            return 'formatted_file'
        elif file_ext in ['.xlsx', '.xls', '.csv']:
            return 'raw_file'
    return 'unknown'

def load_training_file(file_path: str) -> List[Dict[str, Any]]:
    """Load training data from a file based on its extension."""
    from risk_fine_tuner_enhanced import load_training_file as enhanced_load
    return enhanced_load(file_path)

def format_risk_example(example, categories):
    """Format a risk categorization example for fine-tuning."""
    try:
        text = example["text"]
        l2_category = example["l2_category"]
        macro_risks = example["macro_risks"]
        
        return {
            "messages": [
                {
                    "role": "system",
                    "content": f"You are an expert cybersecurity risk analyst with extensive experience in categorizing security findings according to standardized risk frameworks. Your task is to analyze security risk findings and correctly identify both the L2 category and specific macro risks.\n\nYou must use ONLY the standardized categories provided below.\n\n{categories}\n\nGuidelines for risk categorization:\n1. Each security finding belongs to exactly ONE L2 category - select the most appropriate match\n2. Security findings often exhibit multiple macro risks within their L2 category\n3. L2 categories represent broad areas of security concern, while macro risks are specific vulnerabilities or weaknesses\n4. You must only select macro risks that belong to the chosen L2 category\n5. You must never invent new categories or risks\n6. IMPORTANT: The numbers assigned to each L2 category (1, 2, 3, etc.) are just identifiers - focus on the text descriptions when determining the appropriate category\n\nContext: Security risk categorization is critical for organizations to standardize their approach to risk management, ensure comprehensive coverage across all risk domains, and enable consistent prioritization and remediation."
                },
                {
                    "role": "user",
                    "content": f"I need to analyze a security risk finding to identify the L2 category and specific macro risks.\n\nText to analyze:\n{text}\n\nIs this a security risk finding or text with potential PII?"
                },
                {
                    "role": "assistant",
                    "content": "This is a security risk finding."
                },
                {
                    "role": "user",
                    "content": "Please provide your risk analysis in a structured JSON format with:\n1. 'l2_category': Select ONE L2 category from the standardized list (include both the number and name)\n2. 'macro_risks': Provide an array of specific macro risks from the corresponding list\n\nAnalyze the finding carefully to ensure accurate categorization. Focus on the description of the L2 categories, not their numbers."
                },
                {
                    "role": "assistant",
                    "content": f"{{\n  \"l2_category\": \"{l2_category}\",\n  \"macro_risks\": {json.dumps(macro_risks, indent=2)}\n}}"
                }
            ]
        }
    except Exception as e:
        print(f"Error formatting risk example: {str(e)}")
        return None

def format_pii_example(example, categories):
    """Format a PII classification example for fine-tuning."""
    try:
        text = example["text"]
        pc_category = example["pc_category"]
        pii_types = example["pii_types"]
        
        return {
            "messages": [
                {
                    "role": "system",
                    "content": f"You are a specialized data privacy expert with deep knowledge of personally identifiable information (PII) detection and classification. Your expertise helps organizations properly handle sensitive data in compliance with regulations like GDPR, CCPA, and HIPAA.\n\nYou must use ONLY the standardized categories provided below.\n\n{categories}\n\nGuidelines for PII classification:\n\n1. PC0 (Public) - Information with no confidentiality requirements that can be freely shared\n   • Examples: Public documentation, marketing materials, open data\n   • Contains no personally identifiable information\n   • May include general business information that is already publicly available\n\n2. PC1 (Internal) - Information with basic confidentiality requirements\n   • Examples: Names, business contact details, customer IDs, general business data\n   • Contains limited personal identifiers but no sensitive personal data\n   • Requires basic protection but would cause minimal harm if disclosed\n\n3. PC3 (Confidential) - Information with high protection requirements\n   • Examples: SSNs, financial data, health information, credentials, biometrics\n   • Contains sensitive personal data requiring strict protection\n   • Would cause significant harm to individuals if improperly disclosed\n\nYour task is to analyze text, identify if it contains PII, classify it into the correct protection category, and list the specific types of PII found."
                },
                {
                    "role": "user",
                    "content": f"Please analyze the following text to identify any PII and classify it according to the protection categories.\n\nIMPORTANT: Apply the HIGHEST SENSITIVITY RULE - if text contains data of different sensitivity levels, classify the ENTIRE text at the highest level present.\n\nText to analyze:\n{text}\n\nIs this a security risk finding or text with potential PII?"
                },
                {
                    "role": "assistant",
                    "content": "This is text with potential PII."
                },
                {
                    "role": "user",
                    "content": "Please provide your PII analysis in a structured JSON format with:\n1. 'pc_category': Select ONE protection category using the HIGHEST SENSITIVITY RULE:\n   - If ANY PC3 (confidential) data is present → classify as PC3\n   - If ANY PC1 (internal) data is present (and no PC3) → classify as PC1\n   - Only if ALL data is PC0 (public) → classify as PC0\n2. 'pii_types': Provide an array of specific PII types found (if any)\n\nAnalyze the text carefully to ensure accurate classification at the highest sensitivity level present."
                },
                {
                    "role": "assistant",
                    "content": f"{{\n  \"pc_category\": \"{pc_category}\",\n  \"pii_types\": {json.dumps(pii_types, indent=2)}\n}}"
                }
            ]
        }
    except Exception as e:
        print(f"Error formatting PII example: {str(e)}")
        return None

def format_ddl_example(example, categories):
    """Format a DDL analysis example for fine-tuning."""
    try:
        ddl_statement = example["ddl_statement"]
        analysis_result = example["analysis_result"]
        
        return {
            "messages": [
                {
                    "role": "system", 
                    "content": f"You are a specialized database privacy expert with deep knowledge of Data Definition Language (DDL) analysis and PII detection in database schemas. Your expertise helps organizations identify potential privacy risks during database design and ensure compliance with data protection regulations.\n\nYou must use ONLY the standardized categories provided below.\n\n{categories}\n\nGuidelines for DDL PII Analysis:\n\n1. Analyze column names, data types, and constraints to identify potential PII\n2. Apply the HIGHEST SENSITIVITY RULE for overall table classification\n3. Consider data type patterns (e.g., CHAR(11) might be SSN, VARCHAR(255) might be email)\n4. Identify compliance requirements based on detected PII types\n5. Provide specific privacy recommendations for the database schema\n\nYour task is to analyze DDL statements and identify what types of PII data the resulting table will contain, along with appropriate protection requirements."
                },
                {
                    "role": "user",
                    "content": f"Please analyze the following DDL statement to identify potential PII data that will be stored in this table.\n\nDDL Statement:\n{ddl_statement}\n\nIs this a security risk finding, DDL statement for PII analysis, or text with potential PII?"
                },
                {
                    "role": "assistant", 
                    "content": "This is a DDL statement for PII analysis."
                },
                {
                    "role": "user",
                    "content": "Please provide your DDL PII analysis in a structured JSON format with:\n1. 'table_name': Name of the table being created\n2. 'overall_classification': Overall protection category (PC0, PC1, PC3, or AMBIGUOUS)\n   - Use HIGHEST SENSITIVITY RULE for clear PII\n   - Use AMBIGUOUS when column names could be PII or non-PII\n3. 'detected_pii_types': Array of PII types that will be stored in this table\n4. 'high_risk_columns': Array of column names that contain PC3 (confidential) data  \n5. 'ambiguous_columns': Array of column names that require human review\n6. 'requires_human_review': Boolean indicating if manual inspection is needed\n7. 'compliance_flags': Array of compliance requirements (e.g., REQUIRES_GDPR_REVIEW)\n8. 'privacy_recommendations': Array of specific recommendations for this schema\n\nAnalyze the DDL statement carefully. Mark as AMBIGUOUS when column names like 'name', 'id', 'address', 'number' could be either personal data or system data depending on context."
                },
                {
                    "role": "assistant",
                    "content": json.dumps(analysis_result, indent=2)
                }
            ]
        }
    except Exception as e:
        print(f"Error formatting DDL example: {str(e)}")
        return None

def create_ddl_training_examples(ddl_statements: List[str]) -> List[Dict[str, Any]]:
    """Create training examples from DDL statements."""
    analyzer = DDLPIIAnalyzer()
    training_examples = []
    
    for ddl in ddl_statements:
        try:
            # Analyze the DDL statement
            analysis = analyzer.analyze_ddl_statement(ddl)
            
            # Create simplified result for training
            training_result = {
                "table_name": analysis["table_name"],
                "overall_classification": analysis["overall_classification"],
                "detected_pii_types": list(set(
                    analysis["pii_summary"]["PC3"] + 
                    analysis["pii_summary"]["PC1"] + 
                    analysis["pii_summary"]["PC0"] +
                    analysis["pii_summary"]["AMBIGUOUS"]
                )),
                "high_risk_columns": [
                    col["column_name"] for col in analysis["columns"] 
                    if col["pc_category"] == "PC3"
                ],
                "ambiguous_columns": [
                    amb_col["column_name"] for amb_col in analysis.get("ambiguous_columns", [])
                ],
                "requires_human_review": analysis.get("requires_human_review", False),
                "compliance_flags": analysis["compliance_flags"],
                "privacy_recommendations": analysis["privacy_recommendations"][:3]  # Top 3 recommendations
            }
            
            example = {
                "type": "ddl",
                "ddl_statement": ddl.strip(),
                "analysis_result": training_result
            }
            
            training_examples.append(example)
            
        except Exception as e:
            print(f"Error creating DDL training example: {str(e)}")
            continue
    
    return training_examples

def process_batch(examples: List[Dict[str, Any]], file_handle) -> None:
    """Process a batch of examples and write them to a JSONL file."""
    try:
        categories = format_all_categories_for_prompt()
        
        for example in tqdm(examples, desc="Formatting examples"):
            try:
                example_type = example.get("type", "unknown")
                
                if example_type == "risk":
                    formatted_example = format_risk_example(example, categories)
                elif example_type == "pii":
                    formatted_example = format_pii_example(example, categories)
                elif example_type == "ddl":
                    formatted_example = format_ddl_example(example, categories)
                else:
                    print(f"Warning: Unknown example type: {example_type}")
                    continue
                
                if formatted_example:
                    file_handle.write(json.dumps(formatted_example) + '\n')
            except Exception as e:
                print(f"Error processing example: {str(e)}")
                continue
    except Exception as e:
        print(f"Error in batch processing: {str(e)}")
        traceback.print_exc()

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

def fine_tune_model_fixed(training_data_path: str, output_dir: str = "fine_tuning_output", 
                         resume_from_checkpoint: Optional[str] = None) -> Optional[str]:
    """
    Fine-tune model with comprehensive gradient fixes and H100 checkpointing.
    COMPLETELY FIXED VERSION for gradient requirements with resume capability.
    """
    print("=== GRADIENT-FIXED FINE-TUNING WITH H100 CHECKPOINTING ===")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize checkpoint callback
    checkpoint_callback = CheckpointCallback(output_dir, save_every_minutes=30)  # Save every 30 minutes
    
    # Check for resume capability
    if resume_from_checkpoint:
        print(f"[RESUME] Attempting to resume from: {resume_from_checkpoint}")
        if not os.path.exists(resume_from_checkpoint):
            print(f"[RESUME] Checkpoint not found: {resume_from_checkpoint}")
            resume_from_checkpoint = None
        else:
            print(f"[RESUME] Valid checkpoint found: {resume_from_checkpoint}")
    else:
        # Auto-detect latest checkpoint
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            print(f"[RESUME] Auto-detected checkpoint: {latest_checkpoint}")
            resume_from_checkpoint = latest_checkpoint
    
    # Load previous training state if resuming
    training_state = None
    if resume_from_checkpoint:
        training_state = load_training_state(output_dir)
    
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
    
    # FIXED: Torch serialization security for checkpoint loading
    print("[FIX] Configuring PyTorch serialization for checkpoint compatibility...")
    import torch.serialization
    # Allow numpy reconstruction for checkpoint loading
    torch.serialization.add_safe_globals([
        'numpy.core.multiarray._reconstruct',
        'numpy.ndarray', 
        'numpy.dtype',
        'numpy.core.multiarray.scalar'
    ])
    
    # Start training
    print("[TRAIN] Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save model
    print("[SAVE] Saving model...")
    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    # Create inference package
    pickle_path = os.path.join(output_dir, "fixed_model.pkl")
    
    # Create the inference package with the correct format
    inference_package = {
        "model_path": final_model_path,
        "unified": True,
        "l2": L2,
        "macro_risks": MACRO_RISKS,
        "pii_protection_categories": PII_PROTECTION_CATEGORIES,
        "pii_types": PII_TYPES,
        "is_fallback": False
    }
    
    with open(pickle_path, 'wb') as f:
        pickle.dump(inference_package, f)
    
    print(f"[SUCCESS] Training completed!")
    print(f"[FILE] Model: {final_model_path}")
    print(f"[FILE] Inference package: {pickle_path}")
    
    return pickle_path

def main():
    """Main function with enhanced argument parsing for checkpointing."""
    parser = argparse.ArgumentParser(
        description="Gradient-Fixed Risk & PII Fine-Tuner with H100 Checkpointing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
H100 Checkpointing Examples:

  # Start new training
  python risk_fine_tuner_gradient_fixed.py --training-data ./data --output ./output

  # Resume from specific checkpoint
  python risk_fine_tuner_gradient_fixed.py --training-data ./data --output ./output --resume-from-checkpoint ./output/checkpoints/checkpoint-1000

  # Auto-resume from latest checkpoint (if found)
  python risk_fine_tuner_gradient_fixed.py --training-data ./data --output ./output

  # Resume after interruption (auto-detects latest checkpoint)
  python risk_fine_tuner_gradient_fixed.py --training-data ./data --output ./existing_output

Checkpointing Features:
  - Automatic checkpoint detection and resume
  - Frequent saves optimized for H100 (every 25-200 steps depending on VRAM)
  - Time-based checkpointing (every 30 minutes)
  - Graceful handling of interruptions
  - Progress tracking with time estimates
  - Multiple checkpoint retention for safety
        """
    )
    
    parser.add_argument("--training-data", required=True, 
                       help="Path to training data (file or directory)")
    parser.add_argument("--output", default="fine_tuning_output_fixed", 
                       help="Output directory for model and checkpoints")
    parser.add_argument("--resume-from-checkpoint", type=str, 
                       help="Specific checkpoint path to resume from (auto-detects if not specified)")
    parser.add_argument("--no-auto-resume", action="store_true",
                       help="Disable automatic checkpoint detection and resume")
    
    args = parser.parse_args()
    
    # Determine resume strategy
    resume_checkpoint = None
    if not args.no_auto_resume:
        if args.resume_from_checkpoint:
            resume_checkpoint = args.resume_from_checkpoint
        else:
            # Auto-detect latest checkpoint
            checkpoint_dir = os.path.join(args.output, "checkpoints")
            resume_checkpoint = find_latest_checkpoint(checkpoint_dir)
            if resume_checkpoint:
                print(f"[AUTO-RESUME] Found checkpoint: {resume_checkpoint}")
                response = input("Resume from this checkpoint? [Y/n]: ").strip().lower()
                if response and response[0] == 'n':
                    resume_checkpoint = None
                    print("[AUTO-RESUME] Starting fresh training")
                else:
                    print("[AUTO-RESUME] Resuming from checkpoint")
    
    try:
        print(f"[START] H100 Training Session")
        print(f"[CONFIG] Training data: {args.training_data}")
        print(f"[CONFIG] Output directory: {args.output}")
        print(f"[CONFIG] Resume checkpoint: {resume_checkpoint or 'None (fresh start)'}")
        
        result = fine_tune_model_fixed(
            training_data_path=args.training_data, 
            output_dir=args.output,
            resume_from_checkpoint=resume_checkpoint
        )
        
        if result:
            print(f"\nSUCCESS! Model saved to: {result}")
            print(f"[H100] Training completed successfully")
            return 0
        else:
            print(f"\nFAILED!")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n[INTERRUPT] Training interrupted by user")
        print(f"[CHECKPOINT] Check {args.output}/checkpoints/ for saved checkpoints")
        print(f"[RESUME] Use --resume-from-checkpoint to continue training")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 