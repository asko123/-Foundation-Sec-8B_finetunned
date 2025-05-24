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
        
        print(f"Loading model from {model_path} ({'unified' if unified else 'task-specific'} model)")
        
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
        if 'macro_risks' in inference_package:
            categories['macro_risks'] = inference_package['macro_risks']
        
        if 'thematic_risks' in inference_package:
            categories['thematic_risks'] = inference_package['thematic_risks']
            
        if 'pii_protection_categories' in inference_package:
            categories['pii_protection_categories'] = inference_package['pii_protection_categories']
            
        if 'pii_types' in inference_package:
            categories['pii_types'] = inference_package['pii_types']
        
        # For backward compatibility, determine if this is a task-specific model
        if not unified:
            if 'task_type' in inference_package:
                categories['task_type'] = inference_package['task_type']
            elif 'macro_risks' in inference_package and 'pii_protection_categories' not in inference_package:
                categories['task_type'] = 'risk'
            elif 'pii_protection_categories' in inference_package and 'macro_risks' not in inference_package:
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
    categories_text = ""
    
    # Add risk categories if available
    if 'macro_risks' in categories and 'thematic_risks' in categories:
        categories_text += "PART 1: SECURITY RISK CATEGORIES\n\n"
        categories_text += "Standardized Macro Risk Categories:\n"
        for key, value in categories['macro_risks'].items():
            categories_text += f"{key}. {value}\n"
        
        categories_text += "\nThematic Risks for each Macro Risk Category:\n"
        for key, themes in categories['thematic_risks'].items():
            categories_text += f"\n{key}. {categories['macro_risks'][key]}:\n"
            for theme in themes:
                categories_text += f"   - {theme}\n"
    
    # Add PII categories if available
    if 'pii_protection_categories' in categories:
        if categories_text:
            categories_text += "\n\nPART 2: PII PROTECTION CATEGORIES\n\n"
        else:
            categories_text += "PII PROTECTION CATEGORIES\n\n"
            
        categories_text += "PII Protection Categories:\n"
        for key, value in categories['pii_protection_categories'].items():
            categories_text += f"{key}: {value}\n"
        
        if 'pii_types' in categories:
            categories_text += "\nCommon PII Types:\n"
            for pii_type in categories['pii_types']:
                categories_text += f"- {pii_type}\n"
    
    return categories_text

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
            "content": f"You are a dual-specialized expert in both cybersecurity risk analysis and data privacy (PII detection). Your first task is to determine whether the given text represents a security risk finding that needs risk categorization, or text that potentially contains personally identifiable information (PII) requiring privacy classification.\n\nBelow are the standardized categories you must use for reference:\n\n{categories_text}\n\nSecurity risk findings typically describe vulnerabilities, threats, or weaknesses in systems, processes, or controls that could be exploited, lead to unauthorized access, or otherwise impact security objectives.\n\nText with PII typically contains personal identifiers like names, contact information, IDs, or sensitive personal data that requires protection based on privacy regulations."
        },
        {
            "role": "user",
            "content": f"Please analyze the following text and determine if it is a security risk finding that requires risk categorization, or text that potentially contains PII requiring privacy classification.\n\nText to analyze:\n{text}\n\nIs this a security risk finding or text with potential PII?"
        }
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
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
    import torch
    import re
    
    # Format the categories
    categories_text = format_all_categories_for_prompt(categories)
    
    # Create an enhanced prompt for risk analysis
    messages = [
        {
            "role": "system",
            "content": f"You are an expert cybersecurity risk analyst with extensive experience in categorizing security findings according to standardized risk frameworks. Your task is to analyze security risk findings and correctly identify both the macro risk category and specific risk themes.\n\nYou must use ONLY the standardized categories provided below.\n\n{categories_text}\n\nGuidelines for risk categorization:\n1. Each security finding belongs to exactly ONE macro risk category - select the most appropriate match\n2. Security findings often exhibit multiple risk themes within their macro category\n3. Macro categories represent broad areas of security concern, while thematic risks are specific vulnerabilities or weaknesses\n4. You must only select thematic risks that belong to the chosen macro risk category\n5. You must never invent new categories or themes\n6. IMPORTANT: The numbers assigned to each macro risk category (1, 2, 3, etc.) are just identifiers - focus on the text descriptions when determining the appropriate category\n\nContext: Security risk categorization is critical for organizations to standardize their approach to risk management, ensure comprehensive coverage across all risk domains, and enable consistent prioritization and remediation."
        },
        {
            "role": "user",
            "content": f"I need to analyze a security risk finding to identify the macro risk category and specific risk themes.\n\nSecurity Risk Finding:\n{text}\n\nPlease provide your analysis in a structured JSON format with:\n1. 'macro_risk': Select ONE macro risk category from the standardized list (include both the number and name)\n2. 'risk_themes': Provide an array of specific risk themes from the corresponding thematic risks list\n\nAnalyze the finding carefully to ensure accurate categorization. Focus on the description of the macro risk categories, not their numbers."
        }
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
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
                "type": "risk",
                "success": True,
                "macro_risk": result.get("macro_risk", ""),
                "risk_themes": result.get("risk_themes", []),
                "raw_response": response
            }
        except json.JSONDecodeError:
            pass
    
    # If we couldn't parse JSON, return the raw response
    return {
        "type": "risk",
        "success": False,
        "raw_response": response
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
            "content": f"Please analyze the following text to identify any PII and classify it according to the protection categories (PC0, PC1, or PC3).\n\nText to analyze:\n{text}\n\nPlease provide your analysis in a structured JSON format with:\n1. 'pc_category': Select ONE protection category (PC0, PC1, or PC3) based on the sensitivity of any PII present\n2. 'pii_types': Provide an array of specific PII types found (if any)\n\nAnalyze the text carefully to ensure accurate classification."
        }
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
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

def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Use a fine-tuned model for unified risk and PII analysis")
    parser.add_argument("--model", type=str, required=True, help="Path to the fine-tuned model pickle file")
    
    # Create options for text input
    parser.add_argument("--text", type=str, help="Text to analyze (for security risk or PII detection)")
    parser.add_argument("--file", type=str, help="File containing texts to analyze (one per line)")
    
    # For backward compatibility
    parser.add_argument("--risk", type=str, help="Risk finding to categorize (same as --text)")
    parser.add_argument("--risk_file", type=str, help="File containing risk findings (same as --file)")
    parser.add_argument("--pii", type=str, help="Text to analyze for PII (same as --text)")
    parser.add_argument("--pii_file", type=str, help="File containing texts to analyze for PII (same as --file)")
    
    parser.add_argument("--output", type=str, help="Path to save the results (JSON format)")
    
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
                print(f"Macro Risk: {result.get('macro_risk', 'Not identified')}")
                print(f"Risk Themes: {', '.join(result.get('risk_themes', ['None']))}")
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
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
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
            
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            traceback.print_exc()
    
    # Save results if output path is provided
    if args.output and results:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    main() 