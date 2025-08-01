#!/usr/bin/env python3
"""
Quick Fix for Empty Dataset Issue
Patches the process_batch function to provide better validation and error reporting.
"""

import json
import os
from typing import List, Dict, Any
from tqdm import tqdm

def improved_process_batch(examples: List[Dict[str, Any]], file_handle, categories: str) -> Dict[str, int]:
    """
    Improved process_batch function with better error handling and reporting.
    Returns statistics about processing.
    """
    stats = {
        "total_input": len(examples),
        "successful_writes": 0,
        "skipped_unknown_type": 0,
        "skipped_formatting_failed": 0,
        "skipped_missing_fields": 0
    }
    
    try:
        for i, example in enumerate(tqdm(examples, desc="Formatting examples")):
            try:
                example_type = example.get("type", "unknown")
                
                # Auto-detect type if missing
                if example_type == "unknown":
                    if "l2_category" in example and "macro_risks" in example:
                        example_type = "risk"
                        example["type"] = "risk"
                    elif "pc_category" in example and "pii_types" in example:
                        example_type = "pii"
                        example["type"] = "pii"
                    elif "ddl_statement" in example:
                        example_type = "ddl"
                        example["type"] = "ddl"
                
                # Validate required fields
                required_fields = {
                    "risk": ["text", "l2_category", "macro_risks"],
                    "pii": ["text", "pc_category", "pii_types"],
                    "ddl": ["ddl_statement", "analysis_result"]
                }
                
                if example_type in required_fields:
                    missing_fields = [field for field in required_fields[example_type] 
                                    if field not in example or example[field] is None]
                    if missing_fields:
                        print(f"[SKIP] Example {i}: Missing fields {missing_fields} for type {example_type}")
                        stats["skipped_missing_fields"] += 1
                        continue
                
                # Format example
                formatted_example = None
                
                if example_type == "risk":
                    formatted_example = format_risk_example_safe(example, categories)
                elif example_type == "pii":
                    formatted_example = format_pii_example_safe(example, categories)
                elif example_type == "ddl":
                    formatted_example = format_ddl_example_safe(example, categories)
                else:
                    print(f"[SKIP] Example {i}: Unknown type '{example_type}'")
                    stats["skipped_unknown_type"] += 1
                    continue
                
                if formatted_example:
                    file_handle.write(json.dumps(formatted_example) + '\n')
                    stats["successful_writes"] += 1
                else:
                    print(f"[SKIP] Example {i}: Formatting failed for type '{example_type}'")
                    stats["skipped_formatting_failed"] += 1
                    
            except Exception as e:
                print(f"[ERROR] Example {i}: {str(e)}")
                stats["skipped_formatting_failed"] += 1
                continue
                
    except Exception as e:
        print(f"[ERROR] Batch processing failed: {str(e)}")
        raise
    
    return stats

def format_risk_example_safe(example, categories):
    """Safe version of format_risk_example with validation."""
    try:
        # Validate required fields
        if not all(key in example for key in ["text", "l2_category", "macro_risks"]):
            return None
        
        text = str(example["text"]).strip()
        l2_category = str(example["l2_category"]).strip()
        macro_risks = example["macro_risks"]
        
        if not text or not l2_category:
            return None
        
        if isinstance(macro_risks, str):
            macro_risks = [macro_risks]
        elif not isinstance(macro_risks, list):
            macro_risks = []
        
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

def format_pii_example_safe(example, categories):
    """Safe version of format_pii_example with validation."""
    try:
        # Validate required fields
        if not all(key in example for key in ["text", "pc_category", "pii_types"]):
            return None
        
        text = str(example["text"]).strip()
        pc_category = str(example["pc_category"]).strip()
        pii_types = example["pii_types"]
        
        if not text or not pc_category:
            return None
        
        if isinstance(pii_types, str):
            pii_types = [pii_types]
        elif not isinstance(pii_types, list):
            pii_types = []
        
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

def format_ddl_example_safe(example, categories):
    """Safe version of format_ddl_example with validation."""
    try:
        # Validate required fields
        if not all(key in example for key in ["ddl_statement", "analysis_result"]):
            return None
        
        ddl_statement = str(example["ddl_statement"]).strip()
        analysis_result = example["analysis_result"]
        
        if not ddl_statement or not analysis_result:
            return None
        
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

def create_validated_training_files(train_examples: List[Dict[str, Any]], 
                                   eval_examples: List[Dict[str, Any]], 
                                   output_dir: str) -> bool:
    """Create training files with validation and detailed reporting."""
    from risk_fine_tuner_gradient_fixed import format_all_categories_for_prompt
    
    categories = format_all_categories_for_prompt()
    
    # Create training files
    train_file = os.path.join(output_dir, "train_fixed.jsonl")
    eval_file = os.path.join(output_dir, "eval_fixed.jsonl")
    
    print("[DATA] Creating validated training files...")
    
    # Process training data
    print(f"[TRAIN] Processing {len(train_examples)} training examples...")
    with open(train_file, 'w', encoding='utf-8') as f:
        train_stats = improved_process_batch(train_examples, f, categories)
    
    print(f"[TRAIN] Results:")
    print(f"  - Input: {train_stats['total_input']}")
    print(f"  - Written: {train_stats['successful_writes']}")
    print(f"  - Skipped (unknown type): {train_stats['skipped_unknown_type']}")
    print(f"  - Skipped (formatting failed): {train_stats['skipped_formatting_failed']}")
    print(f"  - Skipped (missing fields): {train_stats['skipped_missing_fields']}")
    
    # Process evaluation data
    print(f"[EVAL] Processing {len(eval_examples)} evaluation examples...")
    with open(eval_file, 'w', encoding='utf-8') as f:
        eval_stats = improved_process_batch(eval_examples, f, categories)
    
    print(f"[EVAL] Results:")
    print(f"  - Input: {eval_stats['total_input']}")
    print(f"  - Written: {eval_stats['successful_writes']}")
    print(f"  - Skipped (unknown type): {eval_stats['skipped_unknown_type']}")
    print(f"  - Skipped (formatting failed): {eval_stats['skipped_formatting_failed']}")
    print(f"  - Skipped (missing fields): {eval_stats['skipped_missing_fields']}")
    
    # Validate that we have some successful examples
    total_written = train_stats['successful_writes'] + eval_stats['successful_writes']
    total_input = train_stats['total_input'] + eval_stats['total_input']
    
    if total_written == 0:
        print(f"[ERROR] No examples were successfully processed!")
        print(f"[ERROR] All {total_input} input examples were skipped.")
        return False
    
    success_rate = (total_written / total_input) * 100
    print(f"[SUCCESS] {total_written}/{total_input} examples processed ({success_rate:.1f}% success rate)")
    
    if success_rate < 50:
        print(f"[WARNING] Low success rate ({success_rate:.1f}%). Check your input data format.")
    
    return True

if __name__ == "__main__":
    print("Quick Fix for Empty Dataset Issue")
    print("This module provides improved functions for training data processing.")
    print("Import this module and use create_validated_training_files() instead of the original process_batch().") 