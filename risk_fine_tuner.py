#!/usr/bin/env python3
"""
Unified Risk & PII Fine-Tuner - A tool for fine-tuning the Foundation-Sec-8B model.

This script handles simultaneously training the model for:
1. Security risk categorization
2. PII detection with PC0/PC1/PC3 classification

Enhanced to automatically process raw Excel/CSV files from a folder.
"""

import os
import json
import csv
import random
import pickle
import gc
import traceback
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from tqdm import tqdm
import pandas as pd

# Import enhanced raw data processing functions
from risk_fine_tuner_enhanced import (
    scan_folder_for_files,
    extract_text_from_excel,
    extract_text_from_csv,
    analyze_content_for_risk_category,
    analyze_content_for_pii,
    process_raw_data_to_training_examples,
    process_folder_for_training_data
)

# Define the standardized macro risk categories and thematic risks
MACRO_RISKS = {
    "1": "Operating Model & Risk Management",
    "2": "Develop and Acquire Software and Systems",
    "3": "Manage & Demise IT Assets",
    "4": "Manage Data",
    "5": "Protect Data",
    "6": "Identity & Access Management",
    "7": "Manage Infrastructure",
    "8": "Manage IT Vulnerabilities & Patching",
    "9": "Manage Technology Capacity & Resources",
    "10": "Monitor & Respond to Technology Incidents",
    "11": "Monitor and Respond to Security Incidents",
    "12": "Manage Business Continuity and Disaster Recovery"
}

THEMATIC_RISKS = {
    "1": [
        "Policy/Standard Review",
        "KCI / KRI completeness",
        "IT General & Baseline Controls (Coverage)",
        "Framework Controls (External/Internal)",
        "Exception Management & Risk Tolerance",
        "Issue Management",
        "Monitoring & Testing (MAT)",
        "Security / IT Awareness Training",
        "Maturity Baseline (Yearly)",
        "Governance (Operational Controls)"
    ],
    "2": [
        "Flag Ship Control Coverage",
        "Business Requirement Approval Process",
        "Change Process (Standards & Emergency)",
        "Post Implementation Evaluation (ORE)",
        "Software Dependencies (Internal and External)",
        "M&A – Control Coverage"
    ],
    "3": [
        "Inventory Accuracy & Completeness",
        "Asset Classification & Governance",
        "End of Life – (Hardware and Software)",
        "Asset Destruction (Storage / Media)"
    ],
    "4": [
        "Data Identification, Inventory & Lineage",
        "Data Classification & Governance",
        "Data Quality Controls"
    ],
    "5": [
        "Data Monitoring Processes",
        "Encryption (At Rest, Use, Transit)",
        "Data Loss Prevention",
        "Sensitive Data Logging",
        "Third Party Data Protection",
        "Removable Media"
    ],
    "6": [
        "Authentication",
        "Authorization",
        "Privilege Management",
        "Identity Access Lifecycle (Joiners/Movers/Leavers)",
        "Segregation of Duties",
        "Secrets Management",
        "Production Access"
    ],
    "7": [
        "Configuration Management",
        "Network Segmentation",
        "Cloud Controls",
        "Data Center Management"
    ],
    "8": [
        "Scanning Completeness",
        "Patching Completeness",
        "S-SDLC drafts",
        "Vulnerability assessment and risk treatment"
    ],
    "9": [
        "Capacity Planning",
        "SLO Management",
        "Monitoring (Availability, Performance and Latency)"
    ],
    "10": [
        "Incident Identification & Classification",
        "Tech Incident Reporting & Escalation",
        "Thematic & Trends"
    ],
    "11": [
        "Incident Response Planning",
        "Incident Monitoring and Handling",
        "Security Incident Reporting & Escalation",
        "Audit Logging / Post Mortem",
        "Incident Response Testing",
        "Threat Intelligence"
    ],
    "12": [
        "Operational Resiliency",
        "Cyber Resilience"
    ]
}

# Define PII protection categories
PII_PROTECTION_CATEGORIES = {
    "PC0": "Public information with no confidentiality requirements",
    "PC1": "Internal information with basic confidentiality requirements",
    "PC3": "Confidential information with high protection requirements"
}

# Define common PII types for reference
PII_TYPES = [
    "Name", "Email", "Phone", "Address", "SSN", "Financial", "Health", 
    "Credentials", "Biometric", "National ID", "DOB", "Gender",
    "Location", "IP Address", "Device ID", "Customer ID", "Employment"
]

def format_all_categories_for_prompt() -> str:
    """Format all categories (risk and PII) for inclusion in prompts."""
    categories_text = "PART 1: SECURITY RISK CATEGORIES\n\n"
    categories_text += "Standardized Macro Risk Categories:\n"
    for key, value in MACRO_RISKS.items():
        categories_text += f"{key}. {value}\n"
    
    categories_text += "\nThematic Risks for each Macro Risk Category:\n"
    for key, themes in THEMATIC_RISKS.items():
        categories_text += f"\n{key}. {MACRO_RISKS[key]}:\n"
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
        if file_extension == '.jsonl':
            return load_jsonl_training_data(file_path)
        elif file_extension == '.json':
            return load_json_training_data(file_path)
        elif file_extension in ['.xlsx', '.xls', '.csv']:
            # Process as raw data
            print(f"Processing raw data file: {file_path}")
            if file_extension in ['.xlsx', '.xls']:
                raw_data = extract_text_from_excel(file_path)
            else:
                raw_data = extract_text_from_csv(file_path)
            
            return process_raw_data_to_training_examples(raw_data)
        else:
            print(f"Warning: Unsupported file format: {file_extension}")
            return []
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
    text = example["text"]
    macro_risk = example["macro_risk"]
    risk_themes = example["risk_themes"]
    
    # Format for fine-tuning in Llama chat format with enhanced prompting
    return {
        "messages": [
            {
                "role": "system",
                "content": f"You are an expert cybersecurity risk analyst with extensive experience in categorizing security findings according to standardized risk frameworks. Your task is to analyze security risk findings and correctly identify both the macro risk category and specific risk themes.\n\nYou must use ONLY the standardized categories provided below.\n\n{categories}\n\nGuidelines for risk categorization:\n1. Each security finding belongs to exactly ONE macro risk category - select the most appropriate match\n2. Security findings often exhibit multiple risk themes within their macro category\n3. Macro categories represent broad areas of security concern, while thematic risks are specific vulnerabilities or weaknesses\n4. You must only select thematic risks that belong to the chosen macro risk category\n5. You must never invent new categories or themes\n6. IMPORTANT: The numbers assigned to each macro risk category (1, 2, 3, etc.) are just identifiers - focus on the text descriptions when determining the appropriate category\n\nContext: Security risk categorization is critical for organizations to standardize their approach to risk management, ensure comprehensive coverage across all risk domains, and enable consistent prioritization and remediation."
            },
            {
                "role": "user",
                "content": f"I need to analyze a security risk finding to identify the macro risk category and specific risk themes.\n\nText to analyze:\n{text}\n\nIs this a security risk finding or text with potential PII?"
            },
            {
                "role": "assistant",
                "content": "This is a security risk finding."
            },
            {
                "role": "user",
                "content": "Please provide your risk analysis in a structured JSON format with:\n1. 'macro_risk': Select ONE macro risk category from the standardized list (include both the number and name)\n2. 'risk_themes': Provide an array of specific risk themes from the corresponding thematic risks list\n\nAnalyze the finding carefully to ensure accurate categorization. Focus on the description of the macro risk categories, not their numbers."
            },
            {
                "role": "assistant",
                "content": f"{{\n  \"macro_risk\": \"{macro_risk}\",\n  \"risk_themes\": {json.dumps(risk_themes, indent=2)}\n}}"
            }
        ]
    }

def format_pii_example(example, categories):
    """Format a PII classification example for fine-tuning."""
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

def process_batch(examples: List[Dict[str, Any]], file_handle) -> None:
    """
    Process a batch of examples and write them to a JSONL file.
    
    Args:
        examples: List of training examples
        file_handle: File handle to write to
    """
    categories = format_all_categories_for_prompt()
    
    for example in tqdm(examples, desc="Formatting examples"):
        example_type = example.get("type", "unknown")
        
        if example_type == "risk":
            formatted_example = format_risk_example(example, categories)
        elif example_type == "pii":
            formatted_example = format_pii_example(example, categories)
        else:
            print(f"Warning: Unknown example type: {example_type}")
            continue
        
        file_handle.write(json.dumps(formatted_example) + '\n')

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
            # Import required libraries
            from transformers import (
                AutoModelForCausalLM, 
                AutoTokenizer, 
                TrainingArguments, 
                Trainer, 
                DataCollatorForLanguageModeling
            )
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from datasets import load_dataset
            import torch
            
            print("Loading base model: fdtn-ai/Foundation-Sec-8B")
            
            # Check device availability
            if torch.cuda.is_available():
                print(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
                device = "cuda"
                vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                print(f"Available VRAM: {vram_mb:.2f} MB")
                
                # Adjust batch size based on VRAM
                if vram_mb < 16000:  # Less than 16GB
                    bs = 1
                    ga_steps = 8
                    print("Limited VRAM detected, using smaller batch size")
                else:
                    bs = 2
                    ga_steps = 4
            else:
                print("CUDA not available. Using CPU (this will be slow)")
                device = "cpu"
                bs = 1
                ga_steps = 8
            
            # Load model with quantization for efficiency
            model = AutoModelForCausalLM.from_pretrained(
                "fdtn-ai/Foundation-Sec-8B",
                load_in_8bit=True if device == "cuda" else False,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            
            if device == "cuda":
                model = prepare_model_for_kbit_training(model)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained("fdtn-ai/Foundation-Sec-8B")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Set up chat template
            chat_template = """{% for message in messages %}
{% if message['role'] == 'system' %}{% if not loop.first %}{{ '\\n' }}{% endif %}{{ message['content'] }}
{% elif message['role'] == 'user' %}
User: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}
Assistant: {{ message['content'] }}
{% endif %}
{% endfor %}
Assistant:"""
            tokenizer.chat_template = chat_template

            # Configure LoRA for parameter-efficient fine-tuning
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            
            # Apply LoRA
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            
            # Load datasets
            train_dataset = load_dataset('json', data_files=train_file, split='train')
            eval_dataset = load_dataset('json', data_files=eval_file, split='train')
            
            # Calculate training steps
            num_epochs = 3
            total_steps = len(train_dataset) * num_epochs // (bs * ga_steps)
            eval_steps = max(100, total_steps // 10)
            save_steps = max(200, total_steps // 5)
            
            print(f"Training for {num_epochs} epochs with {total_steps} total steps")
            
            # Prepare data formatting function
            def format_chat(example):
                if isinstance(example["messages"], list):
                    formatted_text = ""
                    for msg in example["messages"]:
                        if msg["role"] == "system":
                            formatted_text += f"{msg['content']}\n"
                        elif msg["role"] == "user":
                            formatted_text += f"User: {msg['content']}\n"
                        elif msg["role"] == "assistant":
                            formatted_text += f"Assistant: {msg['content']}\n"
                    return {"input_ids": tokenizer.encode(formatted_text + "Assistant: ", add_special_tokens=True)}
                return {"input_ids": tokenizer.encode(str(example["messages"]), add_special_tokens=True)}
            
            print("Processing training dataset...")
            train_dataset = train_dataset.map(format_chat, remove_columns=["messages"])
            
            print("Processing evaluation dataset...")
            eval_dataset = eval_dataset.map(format_chat, remove_columns=["messages"])
            
            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=os.path.join(output_dir, "checkpoints"),
                    overwrite_output_dir=True,
                    num_train_epochs=num_epochs,
                    per_device_train_batch_size=bs,
                    per_device_eval_batch_size=bs,
                    gradient_accumulation_steps=ga_steps,
                    learning_rate=2e-5,
                    weight_decay=0.01,
                    warmup_ratio=0.03,
                    logging_steps=10,
                    eval_steps=eval_steps,
                    save_steps=save_steps,
                    eval_strategy="steps",
                    save_strategy="steps",
                    load_best_model_at_end=True,
                    fp16=True if device == "cuda" else False,
                    report_to="none",
                    save_total_limit=3,
                    logging_first_step=True,
                    dataloader_num_workers=4 if device == "cuda" else 0,
                    dataloader_drop_last=True
                ),
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            )
            
            # Start training
            print("Starting training...")
            trainer.train()
            
            # Save the final model
            print("Training complete. Saving the model...")
            final_model_path = os.path.join(output_dir, "final_model")
            trainer.save_model(final_model_path)
            tokenizer.save_pretrained(final_model_path)
            
            # Clean up memory
            del model
            del trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Create the pickle file for inference
            pickle_path = os.path.join(output_dir, "unified_model_with_categories.pkl")
            print(f"Creating inference package at: {pickle_path}")
            
            # Create the inference package with the exact format that risk_inference.py expects
            inference_package = {
                "model_path": final_model_path,
                "unified": True,
                "macro_risks": MACRO_RISKS,
                "thematic_risks": THEMATIC_RISKS,
                "pii_protection_categories": PII_PROTECTION_CATEGORIES,
                "pii_types": PII_TYPES
            }
            
            with open(pickle_path, 'wb') as f:
                pickle.dump(inference_package, f)
            
            print(f"\n✅ Fine-tuning complete!")
            print(f"📁 Fine-tuned model saved to: {final_model_path}")
            print(f"🔗 Inference package saved to: {pickle_path}")
            print(f"\n🚀 To use the model for inference:")
            print(f"   python risk_inference.py --model {pickle_path} --text \"Your text to analyze\"")
            
            return pickle_path
            
        except ImportError as e:
            print(f"\n❌ Missing required packages for fine-tuning: {str(e)}")
            print("Install required packages with:")
            print("pip install transformers datasets accelerate peft bitsandbytes torch")
            return None
            
        except Exception as e:
            print(f"\n❌ Error during fine-tuning: {str(e)}")
            traceback.print_exc()
            
            # Even if fine-tuning fails, create a basic inference package for testing
            print("\nCreating basic inference package for testing...")
            pickle_path = os.path.join(output_dir, "basic_inference_package.pkl")
            
            basic_package = {
                "model_path": "fdtn-ai/Foundation-Sec-8B",  # Use base model
                "unified": True,
                "macro_risks": MACRO_RISKS,
                "thematic_risks": THEMATIC_RISKS,
                "pii_protection_categories": PII_PROTECTION_CATEGORIES,
                "pii_types": PII_TYPES
            }
            
            with open(pickle_path, 'wb') as f:
                pickle.dump(basic_package, f)
                
            print(f"📦 Basic inference package created at: {pickle_path}")
            print("⚠️  This uses the base model (not fine-tuned)")
            return pickle_path
        
    except Exception as e:
        print(f"Fine-tuning failed: {str(e)}")
        traceback.print_exc()
        return None

def main():
    """Main function with enhanced raw data processing capabilities."""
    import argparse
    
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
            print("\n✓ Detected raw data folder - will automatically extract and process Excel/CSV files")
        elif data_format == 'raw_file':
            print("\n✓ Detected raw data file - will extract and process content")
        elif data_format == 'formatted_file':
            print("\n✓ Detected pre-formatted training data file")
        else:
            print(f"\n⚠ Warning: Unknown data format. Will attempt to process anyway.")
        
        # Start fine-tuning
        result = fine_tune_model(args.training_data, args.output)
        
        if result:
            print(f"\n✅ Fine-tuning completed successfully!")
            print(f"Model saved to: {result}")
        else:
            print(f"\n❌ Fine-tuning failed!")
            
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 