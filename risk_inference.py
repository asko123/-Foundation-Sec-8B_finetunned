#!/usr/bin/env python3
"""
Unified Risk & PII Inference - A tool for using a fine-tuned Foundation-Sec-8B model.

This script loads a unified fine-tuned model and automatically detects whether to:
1. Identify standardized macro risk categories and thematic risks for security findings
2. Detect PII and classify data into protection categories (PC0, PC1, PC3)
"""

import os
import json
import traceback
import re
from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm

def load_inference_model(pickle_path: str) -> Tuple:
    """
    Load a fine-tuned model from a pickle file for inference.
    
    Args:
        pickle_path: Path to the pickle file containing the model and categories
        
    Returns:
        Tuple of (model, tokenizer, unified, categories)
    """
    try:
        import pickle
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print(f"Loading inference package from {pickle_path}...")
        with open(pickle_path, 'rb') as f:
            inference_package = pickle.load(f)
        
        model_path = inference_package['model_path']
        unified = inference_package.get('unified', False)
        is_fallback = inference_package.get('is_fallback', False)
        
        print(f"Loading model from {model_path} ({'unified' if unified else 'task-specific'} model)")
        if is_fallback:
            print("WARNING: Using fallback model (fine-tuning was not successful)")
        
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU for inference")
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        # Extract the categories from the package
        categories = {}
        if 'l2' in inference_package:
            categories['l2'] = inference_package['l2']
        elif 'macro_risks' in inference_package:  # For backward compatibility
            categories['l2'] = inference_package['macro_risks']
            
        if 'macro_risks' in inference_package:
            categories['macro_risks'] = inference_package['macro_risks']
        elif 'thematic_risks' in inference_package:  # For backward compatibility
            categories['macro_risks'] = inference_package['thematic_risks']
            
        if 'pii_protection_categories' in inference_package:
            categories['pii_protection_categories'] = inference_package['pii_protection_categories']
            
        if 'pii_types' in inference_package:
            categories['pii_types'] = inference_package['pii_types']
        
        # For backward compatibility, determine if this is a task-specific model
        if not unified:
            if 'task_type' in inference_package:
                categories['task_type'] = inference_package['task_type']
            elif 'l2' in inference_package and 'pii_protection_categories' not in inference_package:
                categories['task_type'] = 'risk'
            elif 'pii_protection_categories' in inference_package and 'l2' not in inference_package:
                categories['task_type'] = 'pii'
            else:
                categories['task_type'] = 'unified'
                unified = True
        
        print("Model loaded successfully")
        return model, tokenizer, unified, categories
        
    except Exception as e:
        print(f"Error loading inference model: {str(e)}")
        traceback.print_exc()
        return None, None, None, None

def format_all_categories_for_prompt(categories: Dict) -> str:
    """Format all categories for inclusion in prompts."""
    text = "PART 1: SECURITY RISK CATEGORIES\n\n"
    
    # Format L2 categories
    if 'l2' in categories:
        text += "L2 Categories:\n"
        for key, value in categories['l2'].items():
            text += f"{key}. {value}\n"
        
        text += "\nMacro Risks for each L2 Category:\n"
        for key, risks in categories['macro_risks'].items():
            text += f"\n{key}. {categories['l2'][key]}:\n"
            for risk in risks:
                text += f"   - {risk}\n"
    
    # Format PII categories
    if 'pii_protection_categories' in categories:
        text += "\n\nPART 2: PII PROTECTION CATEGORIES\n\n"
        text += "PII Protection Categories:\n"
        for key, value in categories['pii_protection_categories'].items():
            text += f"{key}: {value}\n"
        
        if 'pii_types' in categories:
            text += "\nCommon PII Types:\n"
            for pii_type in categories['pii_types']:
                text += f"- {pii_type}\n"
    
    return text

def analyze_text(model, tokenizer, unified: bool, categories: Dict, text: str) -> Dict:
    """
    Analyze text using the fine-tuned model.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer for the model
        unified: Whether this is a unified model
        categories: Dictionary of categories
        text: The text to analyze
        
    Returns:
        Dictionary with analysis results
    """
    try:
        import torch
        
        # For unified models, we need to first determine if this is a risk or PII text
        if unified:
            # First, determine if this is a risk finding or PII text
            text_type = detect_text_type(model, tokenizer, categories, text)
            if text_type == "risk":
                return analyze_risk(model, tokenizer, categories, text)
            else:  # text_type == "pii"
                return analyze_pii(model, tokenizer, categories, text)
        else:
            # For backward compatibility with task-specific models
            task_type = categories.get('task_type', 'risk')
            if task_type == 'risk':
                return analyze_risk(model, tokenizer, categories, text)
            else:  # task_type == 'pii'
                return analyze_pii(model, tokenizer, categories, text)
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }

def format_messages_for_model(messages: List[Dict], tokenizer) -> str:
    """
    Format messages for the model, with fallback when chat template is not available.
    """
    try:
        # Try to use the chat template if available
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception as e:
        print(f"Warning: Chat template not available ({e}), using fallback formatting")
    
    # Fallback to manual formatting
    formatted_text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            formatted_text += f"System: {content}\n\n"
        elif role == "user":
            formatted_text += f"User: {content}\n\n"
        elif role == "assistant":
            formatted_text += f"Assistant: {content}\n\n"
    
    return formatted_text.strip()

def detect_text_type(model, tokenizer, categories: Dict, text: str) -> str:
    """
    Detect whether the text is a security risk finding or text with potential PII.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer for the model
        categories: Dictionary of categories
        text: The text to analyze
        
    Returns:
        String indicating the text type: 'risk' or 'pii'
    """
    import torch
    
    # Format the categories
    categories_text = format_all_categories_for_prompt(categories)
    
    # Create an enhanced prompt to detect the text type
    messages = [
        {
            "role": "system",
            "content": f"You are a triple-specialized expert in cybersecurity risk analysis, data privacy (PII detection), and database schema privacy analysis. Your first task is to determine whether the given input represents:\n1. A security risk finding that needs risk categorization\n2. Text that potentially contains personally identifiable information (PII) requiring privacy classification\n3. A DDL (Data Definition Language) statement that needs database schema privacy analysis\n\nBelow are the standardized categories you must use for reference:\n\n{categories_text}\n\nSecurity risk findings typically describe vulnerabilities, threats, or weaknesses in systems, processes, or controls that could be exploited, lead to unauthorized access, or otherwise impact security objectives.\n\nText with PII typically contains personal identifiers like names, contact information, IDs, or sensitive personal data that requires protection based on privacy regulations.\n\nDDL statements are SQL commands like CREATE TABLE, ALTER TABLE that define database schemas and may indicate what types of PII data will be stored."
        },
        {
            "role": "user",
            "content": f"Please analyze the following input and determine if it is:\n1. A security risk finding that requires risk categorization\n2. Text that potentially contains PII requiring privacy classification\n3. A DDL statement that requires database schema privacy analysis\n\nInput to analyze:\n{text}\n\nIs this a security risk finding, text with potential PII, or a DDL statement?"
        }
    ]
    
    # Create inputs without moving to device (model is already on the correct device)
    formatted_text = format_messages_for_model(messages, tokenizer)
    inputs = tokenizer(formatted_text, return_tensors="pt")
    
    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=50,
            temperature=0.1,
            top_p=0.9,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    # Check if the response indicates this is a risk finding or PII text
    response = response.lower()
    if "security risk" in response or "risk finding" in response or "vulnerability" in response or "categorization" in response:
        return "risk"
    else:
        return "pii"

def analyze_risk(model, tokenizer, categories: Dict, text: str) -> Dict:
    """Analyze text as a security risk finding."""
    try:
        # Format the categories
        categories_text = format_all_categories_for_prompt(categories)
        
        # Create an enhanced prompt for risk analysis
        messages = [
            {
                "role": "system",
                "content": f"You are an expert cybersecurity risk analyst with extensive experience in categorizing security findings according to standardized risk frameworks. Your task is to analyze security risk findings and correctly identify both the L2 category and specific macro risks.\n\nYou must use ONLY the standardized categories provided below.\n\n{categories_text}\n\nGuidelines for risk categorization:\n1. Each security finding belongs to exactly ONE L2 category - select the most appropriate match\n2. Security findings often exhibit multiple macro risks within their L2 category\n3. L2 categories represent broad areas of security concern, while macro risks are specific vulnerabilities or weaknesses\n4. You must only select macro risks that belong to the chosen L2 category\n5. You must never invent new categories or risks\n6. IMPORTANT: The numbers assigned to each L2 category (1, 2, 3, etc.) are just identifiers - focus on the text descriptions when determining the appropriate category\n\nContext: Security risk categorization is critical for organizations to standardize their approach to risk management, ensure comprehensive coverage across all risk domains, and enable consistent prioritization and remediation."
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
            }
        ]
        
        # Generate response
        formatted_text = format_messages_for_model(messages, tokenizer)
        inputs = tokenizer(formatted_text, return_tensors="pt")
        
        import torch
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=512,
                temperature=0.1,
                top_p=0.9,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        try:
            # Parse the JSON response
            result = json.loads(response)
            return {
                "success": True,
                "type": "risk",
                "l2_category": result.get("l2_category"),
                "macro_risks": result.get("macro_risks", [])
            }
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": "Failed to parse model response as JSON",
                "raw_response": response
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def analyze_pii(model, tokenizer, categories: Dict, text: str) -> Dict:
    """Analyze text for PII and protection classification."""
    import torch
    import re
    
    # Format the categories
    categories_text = format_all_categories_for_prompt(categories)
    
    # Create an enhanced prompt for PII analysis
    messages = [
        {
            "role": "system",
            "content": f"You are a specialized data privacy expert with deep knowledge of personally identifiable information (PII) detection and classification. Your expertise helps organizations properly handle sensitive data in compliance with regulations like GDPR, CCPA, and HIPAA.\n\nYou must use ONLY the standardized categories provided below.\n\n{categories_text}\n\nGuidelines for PII classification:\n\n1. PC0 (Public) - Information with no confidentiality requirements that can be freely shared\n   • Examples: Public documentation, marketing materials, open data\n   • Contains no personally identifiable information\n   • May include general business information that is already publicly available\n\n2. PC1 (Internal) - Information with basic confidentiality requirements\n   • Examples: Names, business contact details, customer IDs, general business data\n   • Contains limited personal identifiers but no sensitive personal data\n   • Requires basic protection but would cause minimal harm if disclosed\n\n3. PC3 (Confidential) - Information with high protection requirements\n   • Examples: SSNs, financial data, health information, credentials, biometrics\n   • Contains sensitive personal data requiring strict protection\n   • Would cause significant harm to individuals if improperly disclosed\n\nYour task is to analyze text, identify if it contains PII, classify it into the correct protection category, and list the specific types of PII found."
        },
        {
            "role": "user",
            "content": f"Please analyze the following text to identify any PII and classify it according to the protection categories (PC0, PC1, or PC3).\n\nText to analyze:\n{text}\n\nPlease provide your analysis in a structured JSON format with:\n1. 'pc_category': Select ONE protection category using the HIGHEST SENSITIVITY RULE:\n   - If ANY PC3 (confidential) data is present → classify as PC3\n   - If ANY PC1 (internal) data is present (and no PC3) → classify as PC1\n   - Only if ALL data is PC0 (public) → classify as PC0\n2. 'pii_types': Provide an array of specific PII types found (if any)\n\nAnalyze the text carefully to ensure accurate classification."
        }
    ]
    
    # Create inputs without moving to device
    formatted_text = format_messages_for_model(messages, tokenizer)
    inputs = tokenizer(formatted_text, return_tensors="pt")
    
    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.9,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    # Try to parse JSON from the response
    json_match = re.search(r'({.*?})', response.replace('\n', ' '), re.DOTALL)
    if json_match:
        try:
            json_str = json_match.group(1)
            result = json.loads(json_str)
            return {
                "type": "pii",
                "success": True,
                "pc_category": result.get("pc_category", ""),
                "pii_types": result.get("pii_types", []),
                "raw_response": response
            }
        except json.JSONDecodeError:
            pass
    
    # If we couldn't parse JSON, return the raw response
    return {
        "type": "pii",
        "success": False,
        "raw_response": response
    }

def batch_analyze(model, tokenizer, unified: bool, categories: Dict, texts: List[str]) -> List[Dict]:
    """
    Analyze multiple texts using the fine-tuned model.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer for the model
        unified: Whether this is a unified model
        categories: Dictionary of categories
        texts: List of texts to analyze
        
    Returns:
        List of dictionaries with analysis results
    """
    results = []
    
    for i, text in enumerate(tqdm(texts, desc="Analyzing texts")):
        result = analyze_text(model, tokenizer, unified, categories, text)
        result["input_text"] = text
        results.append(result)
        
        # Print progress periodically
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(texts)} items")
    
    return results

def read_input_file(file_path: str) -> List[str]:
    """
    Read texts from various file formats.
    
    Supports:
    - Plain text files (one text per line)
    - JSONL files (with 'text' field)
    - JSON files (array of objects with 'text' field)
    - CSV files (with header, looks for text/description/content columns)
    - Excel files (.xlsx, .xls with text/description/content columns)
    """
    import pandas as pd
    
    file_ext = os.path.splitext(file_path)[1].lower()
    texts = []
    
    try:
        if file_ext == '.jsonl':
            # Read JSONL file
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            entry = json.loads(line)
                            if isinstance(entry, dict):
                                text = entry.get('text', entry.get('description', entry.get('content')))
                                if text:
                                    texts.append(text)
                        except json.JSONDecodeError:
                            print(f"Warning: Skipping invalid JSON line: {line[:100]}...")
        
        elif file_ext == '.json':
            # Read JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for entry in data:
                        if isinstance(entry, dict):
                            text = entry.get('text', entry.get('description', entry.get('content')))
                            if text:
                                texts.append(text)
                elif isinstance(data, dict):
                    text = data.get('text', data.get('description', data.get('content')))
                    if text:
                        texts.append(text)
        
        elif file_ext in ['.csv']:
            # Read CSV file
            df = pd.read_csv(file_path)
            text_columns = [col for col in df.columns if any(term in col.lower() 
                          for term in ['text', 'description', 'content', 'finding', 'detail'])]
            
            if text_columns:
                primary_col = text_columns[0]
                texts = [str(text) for text in df[primary_col].dropna()]
                print(f"Using column '{primary_col}' from CSV file")
            else:
                print(f"Warning: No text column found in CSV. Available columns: {', '.join(df.columns)}")
        
        elif file_ext in ['.xlsx', '.xls']:
            # Read Excel file
            df = pd.read_excel(file_path)
            text_columns = [col for col in df.columns if any(term in col.lower() 
                          for term in ['text', 'description', 'content', 'finding', 'detail'])]
            
            if text_columns:
                primary_col = text_columns[0]
                texts = [str(text) for text in df[primary_col].dropna()]
                print(f"Using column '{primary_col}' from Excel file")
            else:
                print(f"Warning: No text column found in Excel. Available columns: {', '.join(df.columns)}")
        
        else:
            # Read as plain text file
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
    
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        traceback.print_exc()
        return []
    
    # Remove any empty texts and deduplicate
    texts = list(set(text for text in texts if text.strip()))
    return texts

def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Use a fine-tuned model for unified risk and PII analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single text
  python risk_inference.py --model model.pkl --text "your text here"
  
  # Analyze file (supports multiple formats)
  python risk_inference.py --model model.pkl --file input.txt
  python risk_inference.py --model model.pkl --file data.jsonl
  python risk_inference.py --model model.pkl --file findings.csv
  python risk_inference.py --model model.pkl --file data.xlsx
  
  # Save results to file
  python risk_inference.py --model model.pkl --file input.txt --output results.json
        """
    )
    
    parser.add_argument("--model", type=str, required=True, help="Path to the fine-tuned model pickle file")
    
    # Create options for text input
    parser.add_argument("--text", type=str, help="Text to analyze (for security risk or PII detection)")
    parser.add_argument("--file", type=str, help="File containing texts to analyze (supports txt, json, jsonl, csv, xlsx)")
    
    # For backward compatibility
    parser.add_argument("--risk", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--risk_file", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--pii", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--pii_file", type=str, help=argparse.SUPPRESS)
    
    parser.add_argument("--output", type=str, help="Path to save the results (JSON format)")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of texts to process before showing progress")
    
    args = parser.parse_args()
    
    # Load the model
    model, tokenizer, unified, categories = load_inference_model(args.model)
    if model is None:
        print("Failed to load the model")
        exit(1)
    
    # Determine the text to analyze
    text = args.text or args.risk or args.pii
    file_path = args.file or args.risk_file or args.pii_file
    
    if not (text or file_path):
        print("Please provide text to analyze with --text or a file with --file")
        parser.print_help()
        exit(1)
    
    results = []
    
    # Process individual text
    if text:
        print(f"Analyzing text: {text}")
        result = analyze_text(model, tokenizer, unified, categories, text)
        result["input_text"] = text
        
        # Pretty print the result
        print("\nAnalysis Result:")
        if result.get("type") == "risk":
            if result.get("success", False):
                print(f"Type: Security Risk")
                print(f"L2 Category: {result.get('l2_category', 'Not identified')}")
                print(f"Macro Risks: {', '.join(result.get('macro_risks', ['None']))}")
            else:
                print("Failed to analyze as security risk")
                print(f"Raw response: {result.get('raw_response', '')}")
        else:  # PII analysis
            if result.get("success", False):
                print(f"Type: PII")
                print(f"Protection Category: {result.get('pc_category', 'Not identified')}")
                print(f"PII Types: {', '.join(result.get('pii_types', ['None']))}")
            else:
                print("Failed to analyze for PII")
                print(f"Raw response: {result.get('raw_response', '')}")
        
        results.append(result)
    
    # Process file with multiple texts
    if file_path:
        print(f"Processing file: {file_path}")
        texts = read_input_file(file_path)
        
        if texts:
            print(f"Found {len(texts)} texts to process")
            batch_results = batch_analyze(model, tokenizer, unified, categories, texts)
            results.extend(batch_results)
            
            # Print summary
            risk_count = sum(1 for r in batch_results if r.get("type") == "risk")
            pii_count = sum(1 for r in batch_results if r.get("type") == "pii")
            
            success_count = sum(1 for r in batch_results if r.get("success", False))
            print(f"\nSummary:")
            print(f"- Total texts processed: {len(batch_results)}")
            print(f"- Identified as security risks: {risk_count}")
            print(f"- Identified as containing PII: {pii_count}")
            print(f"- Successfully analyzed: {success_count}")
            
            # Print sample results
            print("\nSample Results:")
            for i, result in enumerate(batch_results[:3]):
                print(f"\nText {i+1}:")
                print(f"Type: {result.get('type', 'Unknown')}")
                if result.get('type') == 'risk':
                    print(f"L2 Category: {result.get('l2_category', 'Not identified')}")
                    print(f"Macro Risks: {', '.join(result.get('macro_risks', ['None']))}")
                else:
                    print(f"Protection Category: {result.get('pc_category', 'Not identified')}")
                    print(f"PII Types: {', '.join(result.get('pii_types', ['None']))}")
        else:
            print("No valid texts found in the file")
    
    # Save results if output path is provided
    if args.output and results:
        try:
            output_dir = os.path.dirname(args.output)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": {
                        "total_texts": len(results),
                        "risk_texts": sum(1 for r in results if r.get("type") == "risk"),
                        "pii_texts": sum(1 for r in results if r.get("type") == "pii"),
                        "successful_analyses": sum(1 for r in results if r.get("success", False))
                    },
                    "results": results
                }, f, indent=2)
            print(f"Results saved to {args.output}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    main() 