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

def get_model_device(model) -> str:
    """Get the device of the model."""
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cpu"  # fallback if model has no parameters

def prepare_inputs_for_model(tokenizer, text: str, model):
    """Prepare tokenized inputs and move them to the same device as the model."""
    inputs = tokenizer(
        text, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
        return_attention_mask=True
    )
    device = get_model_device(model)
    
    # Move all tensors to the model's device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs

def parse_risk_response_as_text(response: str) -> Dict:
    """Parse risk analysis response as plain text when JSON parsing fails."""
    import re
    
    result = {"success": False, "l2_category": None, "macro_risks": []}
    
    # Try to find L2 category patterns
    l2_patterns = [
        r"l2[_\s]*category[:\s]*[\"']?([^\"'\n]+)[\"']?",
        r"category[:\s]*[\"']?(\d+\.?\s*[^\"'\n]+)[\"']?",
        r"(\d+\.?\s*[A-Z][^.\n]+)",
    ]
    
    for pattern in l2_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            result["l2_category"] = match.group(1).strip()
            result["success"] = True
            break
    
    # Try to find macro risks with better patterns
    risk_patterns = [
        r"macro[_\s]*risks?[:\s]*\[([^\]]+)\]",
        r"risks?[:\s]*\[([^\]]+)\]",
        r"risks?[:\s]*[\"']([^\"']+)[\"']",
        r"risk[:\s]*([A-Z][^,\n]+)",
        r"-\s*([A-Z][^,\n]+)",  # Bullet points
        r"\d+\.\s*([A-Z][^,\n]+)",  # Numbered lists
    ]
    
    for pattern in risk_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        for match in matches:
            if ',' in match:
                # Split by commas and clean up
                risks = [risk.strip().strip('"\'') for risk in match.split(',')]
                result["macro_risks"].extend([r for r in risks if r and len(r) > 3])
            else:
                # Single risk
                clean_risk = match.strip().strip('"\'')
                if clean_risk and len(clean_risk) > 3:
                    result["macro_risks"].append(clean_risk)
    
    return result

def parse_pii_response_as_text(response: str) -> Dict:
    """Parse PII analysis response as plain text when JSON parsing fails."""
    import re
    
    result = {"success": False, "pc_category": None, "pii_types": []}
    
    # Try to find PC category with more comprehensive patterns
    pc_patterns = [
        r"pc[_\s]*category[:\s]*[\"']?(PC[0-3])[\"']?",
        r"(PC[0-3])",
        r"protection[_\s]*category[:\s]*[\"']?(PC[0-3])[\"']?",
        r"category[:\s]*[\"']?(PC[0-3])[\"']?",
        r"classification[:\s]*[\"']?(PC[0-3])[\"']?",
    ]
    
    # If no explicit PC category found, infer from content
    if not any(re.search(pattern, response, re.IGNORECASE) for pattern in pc_patterns):
        response_lower = response.lower()
        
        # Check for PC3 indicators (highest priority)
        pc3_indicators = [
            'confidential', 'sensitive', 'ssn', 'social security', 'credit card', 
            'financial', 'medical', 'health', 'password', 'credential', 'biometric'
        ]
        
        # Check for PC1 indicators
        pc1_indicators = [
            'internal', 'personal', 'private', 'name', 'email', 'phone', 
            'address', 'customer', 'employee', 'contact'
        ]
        
        # Check for PC0 indicators (check first for negative indicators)
        pc0_indicators = [
            'no sensitive', 'no confidential', 'no pii', 'no personal', 'public only', 
            'marketing only', 'general information', 'not sensitive', 'not confidential'
        ]
        
        # Check for explicit PC0 terms
        pc0_terms = ['public', 'marketing', 'general', 'open']
        
        # Check PC0 first (negative indicators have priority)
        if any(indicator in response_lower for indicator in pc0_indicators):
            result["pc_category"] = "PC0"
            result["success"] = True
        elif any(indicator in response_lower for indicator in pc3_indicators):
            result["pc_category"] = "PC3"
            result["success"] = True
        elif any(indicator in response_lower for indicator in pc1_indicators):
            result["pc_category"] = "PC1" 
            result["success"] = True
        elif any(term in response_lower for term in pc0_terms):
            result["pc_category"] = "PC0"
            result["success"] = True
    
    for pattern in pc_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            result["pc_category"] = match.group(1).upper()
            result["success"] = True
            break
    
    # Try to find PII types with better patterns
    pii_patterns = [
        r"pii[_\s]*types?[:\s]*\[([^\]]+)\]",
        r"types?[:\s]*\[([^\]]+)\]",
        r"pii[_\s]*types?[:\s]*([^,\n]+)",
        r"types?[:\s]*([^,\n]+)",
        r"including\s+([^,\n]+)",
        r"contains?\s+([^,\n]+)",
        r"-\s*([A-Z][^,\n]+)",  # Bullet points
        r":\s*([A-Z][^,\n]+)",  # Colon followed by types
    ]
    
    # Also look for specific PII type keywords in the response
    pii_keywords = {
        'SSN': ['ssn', 'social security', 'social_security'],
        'Name': ['name', 'names'],
        'Email': ['email', 'e-mail', 'mail'],
        'Phone': ['phone', 'telephone', 'mobile', 'cell'],
        'Address': ['address', 'location'],
        'Financial': ['credit card', 'bank account', 'financial'],
        'Health': ['medical', 'health', 'patient'],
        'Credentials': ['password', 'credential'],
        'National ID': ['passport', 'driver license', 'national id']
    }
    
    # Extract using patterns
    for pattern in pii_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        for match in matches:
            if ',' in match:
                types = [pii_type.strip().strip('"\'') for pii_type in match.split(',')]
                result["pii_types"].extend([t for t in types if t and len(t) > 2])
            else:
                clean_type = match.strip().strip('"\'')
                if clean_type and len(clean_type) > 2:
                    result["pii_types"].append(clean_type)
    
    # Extract using keyword matching
    response_lower = response.lower()
    for pii_type, keywords in pii_keywords.items():
        if any(keyword in response_lower for keyword in keywords):
            if pii_type not in result["pii_types"]:
                result["pii_types"].append(pii_type)
    
    # Clean up and deduplicate results
    valid_pii_types = ['SSN', 'Name', 'Email', 'Phone', 'Address', 'Financial', 'Health', 'Credentials', 'National ID', 'DOB', 'Customer ID']
    cleaned_types = []
    
    for pii_type in result["pii_types"]:
        # Check if it's a valid PII type or contains valid PII type
        pii_type_clean = pii_type.strip()
        if pii_type_clean in valid_pii_types:
            if pii_type_clean not in cleaned_types:
                cleaned_types.append(pii_type_clean)
        else:
            # Check if any valid PII type is mentioned in this text
            for valid_type in valid_pii_types:
                if valid_type.lower() in pii_type_clean.lower() and valid_type not in cleaned_types:
                    cleaned_types.append(valid_type)
    
    result["pii_types"] = cleaned_types
    return result

def format_chat_messages(messages: List[Dict], tokenizer) -> str:
    """
    Format chat messages for models, with fallback for tokenizers without chat templates.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        tokenizer: The tokenizer (may or may not have chat_template)
        
    Returns:
        Formatted string ready for tokenization
    """
    try:
        # Try to use the tokenizer's chat template if available
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception as e:
        print(f"Warning: Could not use chat template: {e}")
    
    # Fallback: manual formatting
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

def format_risk_categories_for_prompt(categories: Dict) -> str:
    """Format only risk categories for risk analysis prompts."""
    text = "L2 Risk Categories:\n"
    
    if 'l2' in categories:
        for key, value in categories['l2'].items():
            text += f"{key}. {value}\n"
        
        text += "\nMacro Risks:\n"
        for key, risks in categories['macro_risks'].items():
            text += f"{key}: " + ", ".join(risks) + "\n"
    
    return text

def format_pii_categories_for_prompt(categories: Dict) -> str:
    """Format only PII categories for PII analysis prompts."""
    text = "PII Protection Categories:\n"
    
    if 'pii_protection_categories' in categories:
        for key, value in categories['pii_protection_categories'].items():
            text += f"{key}: {value}\n"
        
        if 'pii_types' in categories:
            text += "\nCommon PII Types: " + ", ".join(categories['pii_types']) + "\n"
    
    return text

def format_all_categories_for_prompt(categories: Dict) -> str:
    """Format all categories for inclusion in prompts (legacy function)."""
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

def analyze_text(model, tokenizer, unified: bool, categories: Dict, text: str, force_type: str = None) -> Dict:
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
        

        # Determine analysis type
        if force_type:
            # Use forced type
            text_type = force_type
        elif unified:
            # Auto-detect for unified models
            text_type = detect_text_type(model, tokenizer, categories, text)
        else:
            # Use task type for task-specific models
            text_type = categories.get('task_type', 'risk')
        
        # Analyze based on determined type
        if text_type == "risk":
            return analyze_risk(model, tokenizer, categories, text)
        else:  # text_type == "pii"
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
    Simple keyword-based detection for better reliability.
    """
    text_lower = text.lower()
    
    # Security risk keywords
    risk_keywords = [
        'vulnerability', 'exploit', 'attack', 'breach', 'security', 'unauthorized', 
        'malware', 'phishing', 'injection', 'xss', 'csrf', 'authentication',
        'authorization', 'encryption', 'patch', 'firewall', 'intrusion',
        'threat', 'risk', 'compliance', 'audit', 'control', 'access control'
    ]
    
    # PII keywords  
    pii_keywords = [
        'name', 'email', 'phone', 'address', 'ssn', 'social security',
        'credit card', 'bank account', 'passport', 'driver license',
        'customer', 'patient', 'employee', 'personal', 'private',
        'confidential', 'sensitive', 'dob', 'date of birth'
    ]
    
    # Count keyword matches
    risk_score = sum(1 for keyword in risk_keywords if keyword in text_lower)
    pii_score = sum(1 for keyword in pii_keywords if keyword in text_lower)
    
    # Simple heuristic: if text mentions specific personal data, it's likely PII
    if any(word in text_lower for word in ['ssn', 'social security', 'credit card', 'bank account', 'passport']):
        return "pii"
    
    # If it has more risk keywords, classify as risk
    if risk_score > pii_score:
        return "risk"
    elif pii_score > 0:
        return "pii"
    else:
        # Default to risk if unclear
        return "risk"

def analyze_risk(model, tokenizer, categories: Dict, text: str) -> Dict:
    """Analyze text as a security risk finding."""
    try:
        # Format the categories
        categories_text = format_risk_categories_for_prompt(categories)
        
        # Create a simple prompt for risk analysis
        messages = [
            {
                "role": "user",
                "content": f"Categorize this security risk:\n\n{categories_text}\n\nRisk: {text}\n\nReturn only JSON:\n{{\n\"l2_category\": \"8. Manage IT Vulnerabilities & Patching\",\n\"macro_risks\": [\"Vulnerability assessment\", \"Patching Completeness\"]\n}}"
            }
        ]
        
        # Generate response
        formatted_prompt = format_chat_messages(messages, tokenizer)
        

        inputs = prepare_inputs_for_model(tokenizer, formatted_prompt, model)
        
        import torch
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=150,  # Shorter for JSON focus
                temperature=0.1,     # Lower for more deterministic output
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'({.*?})', response.replace('\n', ' '), re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                result = json.loads(json_str)
                return {
                    "success": True,
                    "type": "risk",
                    "l2_category": result.get("l2_category"),
                    "macro_risks": result.get("macro_risks", [])
                }
            else:
                # If no JSON found, try to parse the entire response
                result = json.loads(response)
                return {
                    "success": True,
                    "type": "risk",
                    "l2_category": result.get("l2_category"),
                    "macro_risks": result.get("macro_risks", [])
                }
        except (json.JSONDecodeError, AttributeError):
            # Fallback: Try to extract information manually from text
            fallback_result = parse_risk_response_as_text(response.strip())
            return {
                "success": fallback_result["success"],
                "type": "risk",
                "l2_category": fallback_result.get("l2_category", "Could not determine"),
                "macro_risks": fallback_result.get("macro_risks", []),
                "raw_response": response.strip(),
                "parsing_method": "text_fallback"
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
    categories_text = format_pii_categories_for_prompt(categories)
    
    # Create a simple prompt for PII analysis
    messages = [
        {
            "role": "user",
            "content": f"Analyze for PII:\n\n{categories_text}\n\nText: {text}\n\nReturn only JSON:\n{{\n\"pc_category\": \"PC3\",\n\"pii_types\": [\"SSN\", \"Phone\", \"Name\"]\n}}"
        }
    ]
    
    # Create inputs and move to same device as model
    formatted_prompt = format_chat_messages(messages, tokenizer)
    

    inputs = prepare_inputs_for_model(tokenizer, formatted_prompt, model)
    
    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=150,  # Shorter for JSON focus
            temperature=0.1,     # Lower for more deterministic output
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
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
    
    # If we couldn't parse JSON, try text fallback
    fallback_result = parse_pii_response_as_text(response)
    return {
        "type": "pii",
        "success": fallback_result["success"],
        "pc_category": fallback_result.get("pc_category", "Could not determine"),
        "pii_types": fallback_result.get("pii_types", []),
        "raw_response": response,
        "parsing_method": "text_fallback"
    }

def batch_analyze(model, tokenizer, unified: bool, categories: Dict, texts: List[str], force_type: str = None) -> List[Dict]:
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
        result = analyze_text(model, tokenizer, unified, categories, text, force_type)
        
        # Safety check - ensure result is a valid dictionary
        if result is None:
            result = {
                "success": False,
                "error": "Analysis function returned None",
                "type": "unknown"
            }
        
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
    parser.add_argument("--force-type", type=str, choices=['risk', 'pii'], help="Force analysis as specific type (risk or pii)")
    
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
        if args.force_type:
            print(f"Forcing analysis as: {args.force_type}")
        result = analyze_text(model, tokenizer, unified, categories, text, args.force_type)
        
        # Safety check - ensure result is a valid dictionary
        if result is None:
            result = {
                "success": False,
                "error": "Analysis function returned None",
                "type": "unknown"
            }
        
        result["input_text"] = text
        
        # Pretty print the result
        print("\nAnalysis Result:")
        parsing_method = result.get("parsing_method", "json")
        if parsing_method == "text_fallback":
            print("Note: JSON parsing failed, using text fallback")
            
        if result.get("type") == "risk":
            if result.get("success", False):
                print(f"Type: Security Risk")
                print(f"L2 Category: {result.get('l2_category', 'Not identified')}")
                print(f"Macro Risks: {', '.join(result.get('macro_risks', ['None']))}")
                if parsing_method == "text_fallback":
                    print(f"Parsing Method: Text extraction")
            else:
                print("Failed to analyze as security risk")
                print(f"Raw response: {result.get('raw_response', '')}")
        else:  # PII analysis
            if result.get("success", False):
                print(f"Type: PII")
                print(f"Protection Category: {result.get('pc_category', 'Not identified')}")
                print(f"PII Types: {', '.join(result.get('pii_types', ['None']))}")
                if parsing_method == "text_fallback":
                    print(f"Parsing Method: Text extraction")
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
            if args.force_type:
                print(f"Forcing all analysis as: {args.force_type}")
            batch_results = batch_analyze(model, tokenizer, unified, categories, texts, args.force_type)
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