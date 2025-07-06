#!/usr/bin/env python3
"""
Unified Risk & PII Fine-Tuner - A tool for fine-tuning the Foundation-Sec-8B model.

This script handles simultaneously training the model for:
1. Security risk categorization with L2 and Macro Risk classification
2. PII detection with PC0/PC1/PC3 classification

Enhanced to automatically process raw Excel/CSV files from a folder.
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

def is_raw_data_folder(folder_path: str) -> bool:
    """
    Check if a folder contains raw data files (Excel, CSV) that need processing.
    
    Args:
        folder_path: Path to check
        
    Returns:
        True if folder contains raw data files
    """
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        return False
    
    for file in os.listdir(folder_path):
        file_ext = os.path.splitext(file)[1].lower()
        if file_ext in ['.xlsx', '.xls', '.csv']:
            return True
    
    return False

def detect_data_format(training_data_path: str) -> str:
    """
    Detect whether the provided path contains raw data or pre-formatted training data.
    
    Args:
        training_data_path: Path to data
        
    Returns:
        'raw_folder', 'formatted_file', or 'unknown'
    """
    if os.path.isdir(training_data_path):
        if is_raw_data_folder(training_data_path):
            return 'raw_folder'
        else:
            return 'unknown'
    elif os.path.isfile(training_data_path):
        file_ext = os.path.splitext(training_data_path)[1].lower()
        if file_ext in ['.json', '.jsonl']:
            return 'formatted_file'
        elif file_ext in ['.xlsx', '.xls', '.csv']:
            return 'raw_file'
    
    return 'unknown'

def validate_training_example(example: Dict[str, Any]) -> bool:
    """Validate a training example for required fields and formats."""
    if not isinstance(example, dict):
        return False
        
    if "type" not in example or example["type"] not in ["risk", "pii"]:
        return False
        
    if "text" not in example or not isinstance(example["text"], str):
        return False
        
    if example["type"] == "risk":
        if "l2_category" not in example or not isinstance(example["l2_category"], str):
            return False
        if "macro_risks" not in example or not isinstance(example["macro_risks"], list):
            return False
            
    if example["type"] == "pii":
        if "pc_category" not in example or example["pc_category"] not in ["PC0", "PC1", "PC3"]:
            return False
        if "pii_types" not in example or not isinstance(example["pii_types"], list):
            return False
            
    return True

def load_training_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load training data from a file based on its extension.
    
    Args:
        file_path: Path to the training data file
        
    Returns:
        List of training examples (both risk and PII)
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return []
            
        if file_extension == '.jsonl':
            examples = load_jsonl_training_data(file_path)
        elif file_extension == '.json':
            examples = load_json_training_data(file_path)
        elif file_extension in ['.xlsx', '.xls', '.csv']:
            print(f"Processing raw data file: {file_path}")
            if file_extension in ['.xlsx', '.xls']:
                raw_data = extract_text_from_excel(file_path)
            else:
                raw_data = extract_text_from_csv(file_path)
            examples = process_raw_data_to_training_examples(raw_data)
        else:
            print(f"Warning: Unsupported file format: {file_extension}")
            return []
            
        # Validate examples
        valid_examples = []
        invalid_count = 0
        for example in examples:
            if validate_training_example(example):
                valid_examples.append(example)
            else:
                invalid_count += 1
                
        if invalid_count > 0:
            print(f"Warning: Found {invalid_count} invalid training examples that were skipped")
            
        return valid_examples
        
    except Exception as e:
        print(f"Error loading file {file_path}: {str(e)}")
        traceback.print_exc()
        return []

def load_jsonl_training_data(file_path: str) -> List[Dict[str, Any]]:
    """Load training data from JSONL file."""
    examples = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue
                
                try:
                    example = json.loads(line)
                    examples.append(example)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num + 1}: {e}")
    
    except Exception as e:
        print(f"Error loading JSONL training data: {e}")
    
    return examples

def load_json_training_data(file_path: str) -> List[Dict[str, Any]]:
    """Load training data from JSON file."""
    examples = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            if isinstance(data, list):
                examples = data
            else:
                examples = [data]
    
    except Exception as e:
        print(f"Error loading JSON training data: {e}")
    
    return examples

def format_risk_example(example, categories):
    """Format a risk categorization example for fine-tuning."""
    try:
        text = example["text"]
        l2_category = example["l2_category"]
        macro_risks = example["macro_risks"]
        
        # Format for fine-tuning in Llama chat format with enhanced prompting
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
        
        # Format for fine-tuning in Llama chat format with enhanced prompting
        return {
            "messages": [
                {
                    "role": "system",
                    "content": f"You are a specialized data privacy expert with deep knowledge of personally identifiable information (PII) detection and classification. Your expertise helps organizations properly handle sensitive data in compliance with regulations like GDPR, CCPA, and HIPAA.\n\nYou must use ONLY the standardized categories provided below.\n\n{categories}\n\nGuidelines for PII classification:\n\n1. PC0 (Public) - Information with no confidentiality requirements that can be freely shared\n   • Examples: Public documentation, marketing materials, open data\n   • Contains no personally identifiable information\n   • May include general business information that is already publicly available\n\n2. PC1 (Internal) - Information with basic confidentiality requirements\n   • Examples: Names, business contact details, customer IDs, general business data\n   • Contains limited personal identifiers but no sensitive personal data\n   • Requires basic protection but would cause minimal harm if disclosed\n\n3. PC3 (Confidential) - Information with high protection requirements\n   • Examples: SSNs, financial data, health information, credentials, biometrics\n   • Contains sensitive personal data requiring strict protection\n   • Would cause significant harm to individuals if improperly disclosed\n\nYour task is to analyze text, identify if it contains PII, classify it into the correct protection category, and list the specific types of PII found."
                },
                {
                    "role": "user",
                    "content": f"Please analyze the following text to identify any PII and classify it according to the protection categories.\n\nText to analyze:\n{text}\n\nIs this a security risk finding or text with potential PII?"
                },
                {
                    "role": "assistant",
                    "content": "This is text with potential PII."
                },
                {
                    "role": "user",
                    "content": "Please provide your PII analysis in a structured JSON format with:\n1. 'pc_category': Select ONE protection category (PC0, PC1, or PC3) based on the sensitivity of any PII present\n2. 'pii_types': Provide an array of specific PII types found (if any)\n\nAnalyze the text carefully to ensure accurate classification."
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

def process_batch(examples: List[Dict[str, Any]], file_handle) -> None:
    """
    Process a batch of examples and write them to a JSONL file.
    
    Args:
        examples: List of training examples
        file_handle: File handle to write to
    """
    try:
        categories = format_all_categories_for_prompt()
        
        for example in tqdm(examples, desc="Formatting examples"):
            try:
                example_type = example.get("type", "unknown")
                
                if example_type == "risk":
                    formatted_example = format_risk_example(example, categories)
                elif example_type == "pii":
                    formatted_example = format_pii_example(example, categories)
                else:
                    print(f"Warning: Unknown example type: {example_type}")
                    continue
                
                file_handle.write(json.dumps(formatted_example) + '\n')
            except Exception as e:
                print(f"Error processing example: {str(e)}")
                continue
    except Exception as e:
        print(f"Error in batch processing: {str(e)}")
        traceback.print_exc()

def cleanup_memory(aggressive=False):
    """Clean up memory to prevent leaks."""
    import gc
    import torch
    
    # Standard cleanup
    for _ in range(3):
        gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        
        if aggressive:
            # Aggressive cleanup for low-memory environments
            for _ in range(5):
                gc.collect()
                torch.cuda.empty_cache()
            
            # Reset memory stats
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()
            
            # Force memory defragmentation
            try:
                torch.cuda.memory._record_memory_history(enabled=False)
            except:
                pass
            
            print("[CLEANUP] Aggressive memory cleanup completed")
        
        # Optional memory summary
        if hasattr(torch.cuda, 'memory_summary'):
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            print(f"[MEMORY] GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

def low_memory_training_callback(trainer, logs):
    """Callback for aggressive memory cleanup during training in low-memory environments."""
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
    
    # Clean up memory more frequently for low-memory GPUs
    if vram_gb < 12:
        cleanup_memory(aggressive=True)
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Check memory usage and warn if getting close to limit
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            usage_percent = (allocated / total) * 100
            
            if usage_percent > 85:
                print(f"[WARNING] High GPU memory usage: {usage_percent:.1f}% ({allocated:.2f}/{total:.2f}GB)")
                # Additional emergency cleanup
                torch.cuda.empty_cache()
                gc.collect()

def print_memory_usage(stage=""):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[{stage}] GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.2f}GB")
    else:
        print(f"[{stage}] Using CPU - no GPU memory tracking")
        
def setup_device():
    """Set up and return the appropriate device for training."""
    if not torch.cuda.is_available():
        print("CUDA not available. Using CPU (this will be slow)")
        return "cpu", 1, 8  # device, batch_size, gradient_accumulation_steps
    
    # Get the number of available GPUs
    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} GPU(s)")
    
    if n_gpus > 1:
        # When multiple GPUs are available, explicitly choose the first one
        print("Multiple GPUs detected. Using cuda:0")
        torch.cuda.set_device(0)
        device = "cuda:0"
    else:
        device = "cuda"
    
    # Get VRAM info for the selected device
    vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Using GPU: {gpu_name}")
    print(f"Available VRAM: {vram_mb:.2f} MB")
    
    # Ultra-low memory settings for CUDA environments
    print("[CONFIG] Using ultra-low memory settings optimized for low-memory CUDA")
    
    # Categorize GPU memory and set ultra-conservative settings
    if vram_mb < 8000:  # <8GB (RTX 3060, GTX 1070, etc.)
        bs = 1
        ga_steps = 64  # Very high gradient accumulation
        print(f"[WARNING] Low VRAM GPU detected ({vram_mb:.0f}MB) - using extreme memory conservation")
    elif vram_mb < 12000:  # 8-12GB (RTX 3070, RTX 4060 Ti, etc.)
        bs = 1
        ga_steps = 48
        print(f"[CONFIG] Mid-low VRAM GPU detected ({vram_mb:.0f}MB) - using high memory conservation")
    elif vram_mb < 16000:  # 12-16GB (RTX 3080, RTX 4070, etc.)
        bs = 1
        ga_steps = 32
        print(f"[CONFIG] Standard VRAM GPU detected ({vram_mb:.0f}MB) - using memory conservation")
    elif vram_mb < 24000:  # 16-24GB (RTX 3090, RTX 4080, RTX 4090)
        bs = 1
        ga_steps = 24
        print(f"[CONFIG] High VRAM GPU detected ({vram_mb:.0f}MB) - using moderate memory conservation")
    elif "H100" in gpu_name.upper() or vram_mb > 70000:  # H100 or A100 80GB
        bs = 1
        ga_steps = 16
        print(f"[CONFIG] Enterprise GPU detected ({gpu_name}) - using minimal memory conservation")
    else:  # Everything else
        bs = 1
        ga_steps = 32
        print(f"[CONFIG] Unknown GPU configuration ({vram_mb:.0f}MB) - using conservative settings")
    
    return device, bs, ga_steps

def ensure_tensors_on_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    """Ensure all tensors in the batch are on the specified device."""
    return {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}

def format_chat(example, tokenizer, device):
    """Format chat examples and ensure they're on the correct device."""
    if isinstance(example["messages"], list):
        formatted_text = ""
        for msg in example["messages"]:
            if msg["role"] == "system":
                formatted_text += f"{msg['content']}\n"
            elif msg["role"] == "user":
                formatted_text += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                formatted_text += f"Assistant: {msg['content']}\n"
        input_ids = tokenizer.encode(formatted_text + "Assistant: ", add_special_tokens=True)
    else:
        input_ids = tokenizer.encode(str(example["messages"]), add_special_tokens=True)
    
    # Convert to tensor and move to correct device
    return {"input_ids": torch.tensor(input_ids, device=device)}

def create_inference_package(model_path: str, is_fallback: bool = False) -> Dict[str, Any]:
    """Create the inference package with the correct format."""
    return {
        "model_path": model_path,
        "unified": True,
        "l2": L2,
        "macro_risks": MACRO_RISKS,
        "pii_protection_categories": PII_PROTECTION_CATEGORIES,
        "pii_types": PII_TYPES,
        "is_fallback": is_fallback
    }

def fine_tune_model(training_data_path: str, output_dir: str = "fine_tuning_data") -> Optional[str]:
    """
    Fine-tune the model with the given training data on both risk and PII tasks simultaneously.
    
    Args:
        training_data_path: Path to the training data file or directory
        output_dir: Directory to save the fine-tuned model and related files
        
    Returns:
        Path to the pickle file if successful, None otherwise
    """
    # Load data and prepare for fine-tuning
    print(f"Preparing for unified fine-tuning on both risk categorization and PII detection with data from {training_data_path}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Detect data format and process accordingly
    data_format = detect_data_format(training_data_path)
    print(f"Detected data format: {data_format}")
    
    # Load training data
    try:
        print("Loading and processing training data...")
        training_examples = []
        
        if data_format == 'raw_folder':
            # Process raw data folder using enhanced functionality
            print("Processing raw data folder...")
            temp_training_file = process_folder_for_training_data(training_data_path, os.path.join(output_dir, "extracted_data"))
            training_examples = load_training_file(temp_training_file)
            
        elif data_format == 'raw_file':
            # Process single raw file
            print("Processing raw data file...")
            training_examples = load_training_file(training_data_path)
            
        elif data_format == 'formatted_file':
            # Load pre-formatted training data
            training_examples = load_training_file(training_data_path)
            
        elif os.path.isdir(training_data_path):
            # Process multiple files in directory
            for filename in os.listdir(training_data_path):
                file_path = os.path.join(training_data_path, filename)
                if os.path.isfile(file_path):
                    print(f"Processing file: {filename}")
                    training_examples.extend(load_training_file(file_path))
        else:
            training_examples = load_training_file(training_data_path)
        
        if not training_examples:
            raise ValueError("No valid training examples were found in the provided data.")
        
        # Analyze the distribution of example types
        risk_count = sum(1 for ex in training_examples if ex.get("type") == "risk")
        pii_count = sum(1 for ex in training_examples if ex.get("type") == "pii")
        
        print(f"Found {risk_count} risk categorization examples and {pii_count} PII detection examples")
        
        if risk_count == 0 and pii_count == 0:
            raise ValueError("No valid risk or PII examples found in the data.")
        
        if risk_count == 0:
            print("Warning: No risk categorization examples found. Model will only learn PII detection.")
        
        if pii_count == 0:
            print("Warning: No PII detection examples found. Model will only learn risk categorization.")
        
        # Split data for training and validation
        random.shuffle(training_examples)
        split_idx = int(len(training_examples) * 0.9)
        train_examples = training_examples[:split_idx]
        eval_examples = training_examples[split_idx:]
        
        print(f"Created {len(train_examples)} training examples and {len(eval_examples)} evaluation examples")
        
        # Create training and evaluation files
        train_file = os.path.join(output_dir, "unified_train.jsonl")
        eval_file = os.path.join(output_dir, "unified_eval.jsonl")
        
        print("Creating training data files...")
        with open(train_file, 'w', encoding='utf-8') as f:
            process_batch(train_examples, f)
        
        with open(eval_file, 'w', encoding='utf-8') as f:
            process_batch(eval_examples, f)
        
        # Start the fine-tuning process
        print("\nStarting fine-tuning process...")
        
        try:
            print("Loading base model: fdtn-ai/Foundation-Sec-8B")
            
            # Set up device and get training parameters
            device, bs, ga_steps = setup_device()
            
            # Clean up memory before loading model
            cleanup_memory()
            
            # Detect if H100 for BF16 optimization
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
            use_bf16 = "H100" in gpu_name.upper()
            
            # Determine GPU memory category for ultra-low memory optimization
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
            
            # Ultra-aggressive memory settings for low-memory CUDA
            if device == "cuda":
                if vram_gb < 8:  # <8GB GPU
                    max_memory_limit = f"{int(vram_gb * 0.7)}GB"  # Use only 70% of VRAM
                    use_4bit = True
                    use_8bit = False
                elif vram_gb < 12:  # 8-12GB GPU
                    max_memory_limit = f"{int(vram_gb * 0.75)}GB"  # Use 75% of VRAM
                    use_4bit = False
                    use_8bit = True
                elif vram_gb < 16:  # 12-16GB GPU
                    max_memory_limit = f"{int(vram_gb * 0.8)}GB"  # Use 80% of VRAM
                    use_4bit = False
                    use_8bit = True
                else:  # >16GB GPU
                    max_memory_limit = f"{int(vram_gb * 0.85)}GB"  # Use 85% of VRAM
                    use_4bit = False
                    use_8bit = True
                
                print(f"[CONFIG] GPU Memory: {vram_gb:.1f}GB, Limit: {max_memory_limit}, 4bit: {use_4bit}, 8bit: {use_8bit}")
            else:
                max_memory_limit = None
                use_4bit = False
                use_8bit = False
            
            # Load model with memory-optimized settings
            model_kwargs = {
                "torch_dtype": torch.float16 if device == "cuda" else torch.float32,  # Always use FP16 for CUDA
                "device_map": "auto" if device == "cuda" else None,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
                "use_cache": False,  # Disable KV cache to save memory
            }
            
            # Add quantization settings
            if device == "cuda":
                if use_4bit:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    model_kwargs["quantization_config"] = quantization_config
                    print("[CONFIG] Using 4-bit quantization for extreme memory savings")
                elif use_8bit:
                    model_kwargs["load_in_8bit"] = True
                    print("[CONFIG] Using 8-bit quantization for memory savings")
                
                # Set memory limits
                if max_memory_limit:
                    model_kwargs["max_memory"] = {0: max_memory_limit}
                
                # Enable CPU offloading for very low memory
                if vram_gb < 12:
                    model_kwargs["offload_folder"] = os.path.join(output_dir, "offload")
                    model_kwargs["offload_state_dict"] = True
                    print("[CONFIG] Enabling CPU offloading for low memory GPU")
            
            model = AutoModelForCausalLM.from_pretrained(
                "fdtn-ai/Foundation-Sec-8B",
                **model_kwargs
            )
            print_memory_usage("after_model_loading")
            
            if device == "cuda":
                model = prepare_model_for_kbit_training(model)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained("fdtn-ai/Foundation-Sec-8B")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Set up chat template
            tokenizer.chat_template = '''{% for message in messages %}
{% if message['role'] == 'system' %}{% if not loop.first %}{{ '\n' }}{% endif %}{{ message['content'] }}
{% elif message['role'] == 'user' %}
User: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}
Assistant: {{ message['content'] }}
{% endif %}
{% endfor %}
Assistant: '''

            # Configure LoRA for ultra-low memory efficiency
            if vram_gb < 8:  # <8GB GPU - Extreme memory conservation
                lora_r = 2
                lora_alpha = 4
                target_modules = ["q_proj"]  # Only one module for extreme cases
                print("[CONFIG] Ultra-minimal LoRA for <8GB GPU: r=2, alpha=4, 1 module")
            elif vram_gb < 12:  # 8-12GB GPU - High memory conservation
                lora_r = 4
                lora_alpha = 8
                target_modules = ["q_proj", "v_proj"]  # Two modules
                print("[CONFIG] Minimal LoRA for 8-12GB GPU: r=4, alpha=8, 2 modules")
            elif vram_gb < 16:  # 12-16GB GPU - Moderate memory conservation
                lora_r = 8
                lora_alpha = 16
                target_modules = ["q_proj", "v_proj", "k_proj"]  # Three modules
                print("[CONFIG] Conservative LoRA for 12-16GB GPU: r=8, alpha=16, 3 modules")
            else:  # >16GB GPU - Standard configuration
                lora_r = 8
                lora_alpha = 16
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]  # Four modules
                print("[CONFIG] Standard LoRA for >16GB GPU: r=8, alpha=16, 4 modules")
            
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules
            )
            
            # Apply LoRA
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            print_memory_usage("after_lora_application")
            
            # Load datasets
            train_dataset = load_dataset('json', data_files=train_file, split='train')
            eval_dataset = load_dataset('json', data_files=eval_file, split='train')
            
            # Calculate training steps with memory-aware configuration
            if vram_gb < 8:  # <8GB GPU - Reduce epochs for very low memory
                num_epochs = 2
                eval_batch_size = 1
                save_total_limit = 1  # Keep only 1 checkpoint
                print("[CONFIG] Ultra-low memory: 2 epochs, minimal checkpoints")
            elif vram_gb < 12:  # 8-12GB GPU
                num_epochs = 2
                eval_batch_size = 1
                save_total_limit = 2
                print("[CONFIG] Low memory: 2 epochs, few checkpoints")
            else:  # >12GB GPU
                num_epochs = 3
                eval_batch_size = bs
                save_total_limit = 3
                print("[CONFIG] Standard memory: 3 epochs, normal checkpoints")
            
            total_steps = len(train_dataset) * num_epochs // (bs * ga_steps)
            eval_steps = max(100, total_steps // 10)
            save_steps = max(200, total_steps // 5)
            
            print(f"Training for {num_epochs} epochs with {total_steps} total steps")
            print(f"Using batch_size={bs}, eval_batch_size={eval_batch_size}, gradient_accumulation_steps={ga_steps}")
            print_memory_usage("before_dataset_processing")
            
            # Create a data formatting function closure with device access
            def format_chat_wrapper(example):
                return format_chat(example, tokenizer, device)
            
            print("Processing training dataset...")
            train_dataset = train_dataset.map(
                format_chat_wrapper,
                remove_columns=["messages"],
                load_from_cache_file=False
            )
            
            print("Processing evaluation dataset...")
            eval_dataset = eval_dataset.map(
                format_chat_wrapper,
                remove_columns=["messages"],
                load_from_cache_file=False
            )
            print_memory_usage("after_dataset_processing")
            
            # Memory-optimized training arguments
            training_args = TrainingArguments(
                output_dir=os.path.join(output_dir, "checkpoints"),
                overwrite_output_dir=True,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=bs,
                per_device_eval_batch_size=eval_batch_size,
                gradient_accumulation_steps=ga_steps,
                learning_rate=2e-5,
                weight_decay=0.01,
                warmup_ratio=0.03,
                logging_steps=20,  # Reduce logging frequency to save memory
                eval_steps=eval_steps,
                save_steps=save_steps,
                eval_strategy="steps",
                save_strategy="steps",
                load_best_model_at_end=False if vram_gb < 8 else True,  # Disable for ultra-low memory
                metric_for_best_model=None,  # Disable metric tracking to save memory
                greater_is_better=None,
                # Precision settings
                fp16=True if device == "cuda" else False,  # Always use FP16 for CUDA
                bf16=False,  # Disable BF16 to save memory
                fp16_opt_level="O1",  # Conservative FP16 optimization
                # Memory optimization settings
                gradient_checkpointing=True if device == "cuda" else False,
                dataloader_num_workers=0,  # No workers to save memory
                dataloader_pin_memory=False,  # Disable pin memory
                dataloader_drop_last=True,
                remove_unused_columns=False,
                # Training stability
                max_grad_norm=1.0,
                warmup_steps=max(10, total_steps // 50),  # Minimal warmup
                # Checkpointing
                save_total_limit=save_total_limit,
                save_only_model=True,  # Save only model weights, not optimizer state
                # Logging and monitoring
                logging_first_step=True,
                log_level="warning",  # Reduce log verbosity
                report_to="none",
                # Hardware settings
                no_cuda=device == "cpu",
                local_rank=-1,
                ddp_find_unused_parameters=False,
                # Advanced memory settings
                skip_memory_metrics=True,  # Skip memory metrics to save overhead
                log_on_each_node=False,
                ignore_data_skip=True,
                prediction_loss_only=True,  # Only compute loss, no other metrics
                # Optimizer settings
                optim="adamw_torch" if vram_gb < 12 else "adamw_torch_fused",  # Use regular AdamW for low memory
                adam_epsilon=1e-8,
                max_steps=-1,
                # Additional memory optimizations
                push_to_hub=False,
                hub_model_id=None,
                hub_strategy="every_save",
                hub_token=None,
                # Evaluation settings
                eval_strategy="steps" if eval_batch_size > 0 else "no",
                eval_delay=0,
                eval_accumulation_steps=None,
                # Memory-specific optimizations
                length_column_name=None,  # Disable length grouping
                group_by_length=False,  # Disable grouping to save memory
                past_index=-1,
                run_name=None,
                disable_tqdm=False,
                seed=42,
                data_seed=None
            )
            
            # Initialize trainer with memory-optimized settings
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset if eval_batch_size > 0 else None,
                data_collator=DataCollatorForLanguageModeling(
                    tokenizer=tokenizer, 
                    mlm=False,
                    return_tensors="pt",
                    pad_to_multiple_of=8 if device == "cuda" else None
                ),
            )
            
            # Add training callback for low-memory environments
            from transformers import TrainerCallback
            
            class LowMemoryCallback(TrainerCallback):
                def __init__(self, vram_gb):
                    self.vram_gb = vram_gb
                    self.step_count = 0
                
                def on_step_end(self, args, state, control, **kwargs):
                    self.step_count += 1
                    # Clean up memory every few steps for low-memory GPUs
                    if self.vram_gb < 8 and self.step_count % 5 == 0:
                        cleanup_memory(aggressive=True)
                    elif self.vram_gb < 12 and self.step_count % 10 == 0:
                        cleanup_memory(aggressive=True)
                    elif self.vram_gb < 16 and self.step_count % 20 == 0:
                        cleanup_memory(aggressive=False)
                
                def on_evaluate(self, args, state, control, **kwargs):
                    # Always clean up after evaluation
                    if self.vram_gb < 16:
                        cleanup_memory(aggressive=True)
                
                def on_save(self, args, state, control, **kwargs):
                    # Clean up after saving checkpoints
                    cleanup_memory(aggressive=True)
            
            # Add callback for low-memory management
            if vram_gb < 16:
                trainer.add_callback(LowMemoryCallback(vram_gb))
                print(f"[CONFIG] Added low-memory callback for {vram_gb:.1f}GB GPU")
            
            # Start training with ultra-aggressive memory monitoring
            print("Starting low-memory optimized training...")
            print_memory_usage("before_training_start")
            
            # Ultra-aggressive memory cleanup before training
            cleanup_memory(aggressive=True)
            
            # Try training with comprehensive memory monitoring
            try:
                trainer.train()
            except torch.cuda.OutOfMemoryError as e:
                print(f"[ERROR] CUDA OOM Error during training: {e}")
                print("[RECOVERY] Attempting emergency memory recovery...")
                
                # Emergency memory cleanup
                cleanup_memory(aggressive=True)
                
                # Ultra-conservative emergency settings
                print("[RECOVERY] Switching to ultra-conservative emergency settings...")
                trainer.args.per_device_train_batch_size = 1
                trainer.args.gradient_accumulation_steps = max(64, trainer.args.gradient_accumulation_steps * 4)
                trainer.args.dataloader_num_workers = 0
                trainer.args.dataloader_pin_memory = False
                trainer.args.eval_strategy = "no"  # Disable evaluation
                trainer.args.save_strategy = "no"  # Disable checkpointing
                trainer.args.logging_steps = 50  # Reduce logging
                
                # Remove evaluation dataset to save memory
                trainer.eval_dataset = None
                
                # Final memory cleanup and retry
                cleanup_memory(aggressive=True)
                print("[RECOVERY] Retrying training with emergency settings...")
                trainer.train()
            except Exception as e:
                print(f"[ERROR] Training failed with error: {e}")
                cleanup_memory(aggressive=True)
                raise
            
            # Save the final model
            print("Training complete. Saving the model...")
            final_model_path = os.path.join(output_dir, "final_model")
            trainer.save_model(final_model_path)
            tokenizer.save_pretrained(final_model_path)
            
            # Clean up memory
            del model
            del trainer
            cleanup_memory()
            
            # Create the inference package
            pickle_path = os.path.join(output_dir, "unified_model_with_categories.pkl")
            print(f"Creating inference package at: {pickle_path}")
            
            inference_package = create_inference_package(final_model_path)
            with open(pickle_path, 'wb') as f:
                pickle.dump(inference_package, f)
            
            print(f"\n[SUCCESS] Fine-tuning complete!")
            print(f"[FILE] Fine-tuned model saved to: {final_model_path}")
            print(f"[FILE] Inference package saved to: {pickle_path}")
            print(f"\n[INFO] To use the model for inference:")
            print(f"   python risk_inference.py --model {pickle_path} --text \"Your text to analyze\"")
            
            return pickle_path
            
        except ImportError as e:
            print(f"\n[ERROR] Missing required packages for fine-tuning: {str(e)}")
            print("Install required packages with:")
            print("pip install transformers datasets accelerate peft bitsandbytes torch")
            return None
            
        except Exception as e:
            cleanup_memory()  # Clean up memory on error
            print(f"\n[ERROR] Error during fine-tuning: {str(e)}")
            traceback.print_exc()
            return None
            
    except Exception as e:
        cleanup_memory()  # Clean up memory on error
        print(f"Fine-tuning failed: {str(e)}")
        traceback.print_exc()
        return None

def main():
    """Main function with enhanced raw data processing capabilities."""
    parser = argparse.ArgumentParser(description="Enhanced Risk & PII Fine-Tuner")
    parser.add_argument("--training-data", required=True, 
                       help="Path to training data (can be a file or folder with raw Excel/CSV files)")
    parser.add_argument("--output", default="fine_tuning_output", 
                       help="Output directory for fine-tuned model")
    
    args = parser.parse_args()
    
    try:
        print("=== ENHANCED RISK & PII FINE-TUNER ===")
        print(f"Input: {args.training_data}")
        print(f"Output: {args.output}")
        
        # Detect and process the data
        data_format = detect_data_format(args.training_data)
        
        if data_format == 'raw_folder':
            print("\n[OK] Detected raw data folder - will automatically extract and process Excel/CSV files")
        elif data_format == 'raw_file':
            print("\n[OK] Detected raw data file - will extract and process content")
        elif data_format == 'formatted_file':
            print("\n[OK] Detected pre-formatted training data file")
        else:
            print(f"\n[WARNING] Warning: Unknown data format. Will attempt to process anyway.")
        
        # Start fine-tuning
        result = fine_tune_model(args.training_data, args.output)
        
        if result:
            print(f"\n[SUCCESS] Fine-tuning completed successfully!")
            print(f"Model saved to: {result}")
            return 0
        else:
            print(f"\n[ERROR] Fine-tuning failed!")
            return 1
            
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return 1

def process_privacy_data(raw_data: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Process raw privacy data into training examples.
    
    Expected columns:
    - COLUMNNAME: Name of the database column
    - PRIVACYTYPE: Type of privacy data
    - PRIVACYTYPENAME: Name of the privacy type
    - PRIVACYTYPEDESCRIPTION: Description of the privacy type
    - PRIVACYTYPECLASSIFICATION: Privacy classification level
    """
    training_examples = []
    
    try:
        for _, row in raw_data.iterrows():
            # Create context from available fields
            context = f"Database Column: {row['COLUMNNAME']}\n"
            if pd.notna(row['ENTITYNAME']):
                context += f"Entity: {row['ENTITYNAME']}\n"
            if pd.notna(row['ENTITYDESC']):
                context += f"Description: {row['ENTITYDESC']}\n"
            if pd.notna(row['PRIVACYTYPEDESCRIPTION']):
                context += f"Privacy Description: {row['PRIVACYTYPEDESCRIPTION']}\n"
            
            # Determine privacy classification
            privacy_class = "PC1"  # Default to Internal
            if pd.notna(row['PRIVACYTYPECLASSIFICATION']):
                classification = str(row['PRIVACYTYPECLASSIFICATION']).upper()
                if "PUBLIC" in classification or "DP10" in classification:
                    privacy_class = "PC0"
                elif any(level in classification for level in ["CONFIDENTIAL", "DP30", "HIGHLY RESTRICTED"]):
                    privacy_class = "PC3"
            
            # Determine PII types
            pii_types = set()  # Use set to avoid duplicates
            
            # Check PRIVACYTYPENAME
            if pd.notna(row['PRIVACYTYPENAME']):
                ptype = str(row['PRIVACYTYPENAME']).upper()
                if any(term in ptype for term in ["SSN", "TAX ID", "GOVT_ID"]):
                    pii_types.add("SSN")
                if "NAME" in ptype:
                    pii_types.add("Name")
                if "EMAIL" in ptype:
                    pii_types.add("Email")
                if "PHONE" in ptype:
                    pii_types.add("Phone")
                if "ADDRESS" in ptype:
                    pii_types.add("Address")
                if any(term in ptype for term in ["DOB", "BIRTH", "DATE OF BIRTH"]):
                    pii_types.add("DOB")
                if any(term in ptype for term in ["NATIONAL ID", "GOVT_ID", "GOVERNMENT ID"]):
                    pii_types.add("National ID")
                if any(term in ptype for term in ["FINANCIAL", "ACCOUNT", "CREDIT", "DEBIT"]):
                    pii_types.add("Financial")
                if any(term in ptype for term in ["HEALTH", "MEDICAL"]):
                    pii_types.add("Health")
            
            # Check COLUMNNAME for additional hints
            column = str(row['COLUMNNAME']).upper()
            if "SSN" in column or "TAX_ID" in column:
                pii_types.add("SSN")
            if any(term in column for term in ["FIRST_NAME", "LAST_NAME", "FULL_NAME"]):
                pii_types.add("Name")
            if "EMAIL" in column:
                pii_types.add("Email")
            if "PHONE" in column:
                pii_types.add("Phone")
            if "ADDRESS" in column:
                pii_types.add("Address")
            if "DOB" in column or "BIRTH_DATE" in column:
                pii_types.add("DOB")
            
            # Create training example
            example = {
                "type": "pii",
                "text": context,
                "pc_category": privacy_class,
                "pii_types": list(pii_types),  # Convert set back to list
                "metadata": {
                    "source_column": row['COLUMNNAME'],
                    "privacy_type": row['PRIVACYTYPE'] if pd.notna(row['PRIVACYTYPE']) else None,
                    "sensitivity": row['PRIVACYTYPECLASSIFICATION'] if pd.notna(row['PRIVACYTYPECLASSIFICATION']) else None,
                    "entity_type": row['ENTITYTYPE'] if pd.notna(row['ENTITYTYPE']) else None,
                    "entity_name": row['ENTITYNAME'] if pd.notna(row['ENTITYNAME']) else None
                }
            }
            
            training_examples.append(example)
            
    except Exception as e:
        print(f"Error processing privacy data: {str(e)}")
        traceback.print_exc()
    
    return training_examples

def process_findings_data(raw_data: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Process raw findings data into training examples.
    
    Expected columns:
    - Finding_Title: Title of the finding
    - Finding_Description: Detailed description of the finding
    - L2: L2 category
    - macro_risks: List of macro risks
    """
    training_examples = []
    
    try:
        for _, row in raw_data.iterrows():
            # Create context from title and description
            context = f"Finding: {row['Finding_Title']}\n\n"
            if pd.notna(row['Finding_Description']):
                context += f"Description: {row['Finding_Description']}"
            
            # Parse L2 category
            l2_category = None
            if pd.notna(row['L2']):
                l2_value = str(row['L2']).strip()
                # First try exact match
                for key, value in L2.items():
                    if l2_value == value:
                        l2_category = f"{key}. {value}"
                        break
                
                # If no exact match, try fuzzy match
                if not l2_category:
                    for key, value in L2.items():
                        if l2_value in value or value in l2_value:
                            l2_category = f"{key}. {value}"
                            break
            
            # Parse macro risks
            macro_risks = set()  # Use set to avoid duplicates
            if pd.notna(row['macro_risks']):
                # Handle different formats (string, list, comma-separated)
                risks = row['macro_risks']
                if isinstance(risks, str):
                    risks = [r.strip() for r in risks.split(',')]
                elif isinstance(risks, list):
                    risks = [str(r).strip() for r in risks]
                
                # Match with defined macro risks
                for risk in risks:
                    risk = risk.strip()
                    # First try exact match
                    for themes in MACRO_RISKS.values():
                        if risk in themes:
                            macro_risks.add(risk)
                            break
                    
                    # If no exact match, try fuzzy match
                    if not any(risk in macro_risks):
                        for themes in MACRO_RISKS.values():
                            for theme in themes:
                                if risk in theme or theme in risk:
                                    macro_risks.add(theme)
                                    break
            
            # Only create example if we have valid L2 and macro risks
            if l2_category and macro_risks:
                example = {
                    "type": "risk",
                    "text": context,
                    "l2_category": l2_category,
                    "macro_risks": list(macro_risks),  # Convert set back to list
                    "metadata": {
                        "finding_title": row['Finding_Title'],
                        "original_l2": row['L2'] if pd.notna(row['L2']) else None,
                        "original_risks": row['macro_risks'] if pd.notna(row['macro_risks']) else None
                    }
                }
                training_examples.append(example)
            else:
                print(f"Warning: Skipping finding '{row['Finding_Title']}' due to missing L2 category or macro risks")
            
    except Exception as e:
        print(f"Error processing findings data: {str(e)}")
        traceback.print_exc()
    
    return training_examples

def extract_text_from_excel(file_path: str) -> List[Dict[str, Any]]:
    """Enhanced function to extract text from Excel files."""
    try:
        print(f"Reading Excel file: {file_path}")
        
        # Try to read all sheets
        all_sheets = pd.read_excel(file_path, sheet_name=None)
        training_examples = []
        
        for sheet_name, df in all_sheets.items():
            print(f"Processing sheet: {sheet_name}")
            
            # Clean column names
            df.columns = [str(col).strip().upper() for col in df.columns]
            
            # Determine the type of data based on columns
            if any(col in df.columns for col in ['COLUMNNAME', 'PRIVACYTYPE', 'PRIVACYTYPENAME']):
                print(f"Found privacy data in sheet {sheet_name}")
                training_examples.extend(process_privacy_data(df))
            elif any(col in df.columns for col in ['FINDING_TITLE', 'FINDING_DESCRIPTION']):
                print(f"Found findings data in sheet {sheet_name}")
                training_examples.extend(process_findings_data(df))
            else:
                print(f"Warning: Unknown data format in sheet {sheet_name}")
        
        if not training_examples:
            print(f"Warning: No valid training examples found in {file_path}")
        else:
            print(f"Extracted {len(training_examples)} training examples from {file_path}")
        
        return training_examples
            
    except Exception as e:
        print(f"Error processing Excel file {file_path}: {str(e)}")
        traceback.print_exc()
        return []

def extract_text_from_csv(file_path: str) -> List[Dict[str, Any]]:
    """Enhanced function to extract text from CSV files."""
    try:
        print(f"Reading CSV file: {file_path}")
        
        # Try to read with different encodings
        encodings = ['utf-8', 'latin1', 'iso-8859-1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise ValueError(f"Could not read CSV file with any of the attempted encodings: {encodings}")
        
        # Clean column names
        df.columns = [str(col).strip().upper() for col in df.columns]
        
        # Determine the type of data based on columns
        if any(col in df.columns for col in ['COLUMNNAME', 'PRIVACYTYPE', 'PRIVACYTYPENAME']):
            print("Found privacy data")
            training_examples = process_privacy_data(df)
        elif any(col in df.columns for col in ['FINDING_TITLE', 'FINDING_DESCRIPTION']):
            print("Found findings data")
            training_examples = process_findings_data(df)
        else:
            print(f"Warning: Unknown data format in {file_path}")
            return []
        
        if not training_examples:
            print(f"Warning: No valid training examples found in {file_path}")
        else:
            print(f"Extracted {len(training_examples)} training examples from {file_path}")
        
        return training_examples
            
    except Exception as e:
        print(f"Error processing CSV file {file_path}: {str(e)}")
        traceback.print_exc()
        return []

if __name__ == "__main__":
    sys.exit(main())                                    