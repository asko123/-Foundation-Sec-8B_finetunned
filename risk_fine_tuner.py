#!/usr/bin/env python3
"""
Unified Risk & PII Fine-Tuner - A tool for fine-tuning the Foundation-Sec-8B model.

This script handles simultaneously training the model for:
1. Security risk categorization
2. PII detection with PC0/PC1/PC3 classification
"""

import os
import json
import csv
import random
import pickle
import gc
import traceback
from typing import List, Dict, Any, Optional, Tuple, Union
from tqdm import tqdm
import pandas as pd

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

def identify_data_type(file_path: str) -> str:
    """
    Identify if a file contains risk data, PII data, or both.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        String indicating the data type: 'risk', 'pii', or 'mixed'
    """
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.csv':
            # Check a sample of the CSV file
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader)
                
                # Try to read a few rows
                rows = []
                for _ in range(5):
                    try:
                        rows.append(next(reader))
                    except StopIteration:
                        break
                
                if not rows:
                    return 'unknown'
                
                # Check for PC0/PC1/PC3 in second column
                has_pc_categories = any('PC' in row[1].upper() if len(row) > 1 else False for row in rows)
                
                # Check for risk categories in second column
                has_risk_categories = any(any(risk.lower() in row[1].lower() if len(row) > 1 else False 
                                         for risk in MACRO_RISKS.values()) for row in rows)
                
                if has_pc_categories and has_risk_categories:
                    return 'mixed'
                elif has_pc_categories:
                    return 'pii'
                elif has_risk_categories:
                    return 'risk'
                else:
                    # Check header names as a fallback
                    if any('pii' in h.lower() or 'pc' in h.lower() for h in headers):
                        return 'pii'
                    elif any('risk' in h.lower() or 'security' in h.lower() for h in headers):
                        return 'risk'
                    return 'unknown'
                
        elif file_extension in ('.json', '.jsonl'):
            # Read a sample of the JSON file
            if file_extension == '.jsonl':
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = [f.readline() for _ in range(5) if f.readline()]
                    samples = [json.loads(line) for line in lines if line.strip()]
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        samples = data[:5]
                    else:
                        samples = [data]
            
            if not samples:
                return 'unknown'
                
            # Check keys in samples
            has_pc_categories = any('pc_category' in sample for sample in samples)
            has_risk_categories = any('macro_risk' in sample for sample in samples)
            
            if has_pc_categories and has_risk_categories:
                return 'mixed'
            elif has_pc_categories:
                return 'pii'
            elif has_risk_categories:
                return 'risk'
            else:
                return 'unknown'
        
        elif file_extension == '.xlsx':
            # For Excel files, we'll read the first sheet
            df = pd.read_excel(file_path, nrows=5)
            if df.empty:
                return 'unknown'
            
            # Check column headers
            headers = df.columns.tolist()
            has_pc_keywords = any('pc' in str(h).lower() or 'pii' in str(h).lower() for h in headers)
            has_risk_keywords = any('risk' in str(h).lower() or 'security' in str(h).lower() for h in headers)
            
            if has_pc_keywords and has_risk_keywords:
                return 'mixed'
            elif has_pc_keywords:
                return 'pii'
            elif has_risk_keywords:
                return 'risk'
            
            # Check content of second column if it exists
            if len(df.columns) > 1:
                second_col = df.iloc[:, 1].astype(str)
                has_pc_values = any('PC' in val.upper() for val in second_col)
                has_risk_values = any(any(risk.lower() in val.lower() for risk in MACRO_RISKS.values()) for val in second_col)
                
                if has_pc_values and has_risk_values:
                    return 'mixed'
                elif has_pc_values:
                    return 'pii'
                elif has_risk_values:
                    return 'risk'
            
            return 'unknown'
        
        return 'unknown'
    
    except Exception as e:
        print(f"Error identifying data type for {file_path}: {str(e)}")
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
        data_type = identify_data_type(file_path)
        print(f"Identified file {file_path} as {data_type} data")
        
        if file_extension == '.csv':
            return load_csv_training_data(file_path, data_type)
        elif file_extension == '.json' or file_extension == '.jsonl':
            return load_json_training_data(file_path, data_type)
        elif file_extension == '.xlsx':
            return load_excel_training_data(file_path, data_type)
        else:
            print(f"Warning: Unsupported file format: {file_extension}")
            return []
    except Exception as e:
        print(f"Error loading file {file_path}: {str(e)}")
        traceback.print_exc()
        return []

def load_csv_training_data(file_path: str, data_type: str) -> List[Dict[str, Any]]:
    """
    Load training data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        data_type: 'risk', 'pii', 'mixed', or 'unknown'
        
    Returns:
        List of training examples
    """
    examples = []
    line_count = 0
    valid_count = 0
    errors = 0
    
    try:
        # Count total lines for progress reporting
        with open(file_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f) - 1  # Subtract header
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)  # Skip header row
            
            print(f"Processing CSV file with {total_lines} rows...")
            
            for row in tqdm(reader, total=total_lines, desc="Processing CSV"):
                line_count += 1
                try:
                    if len(row) < 2:
                        continue  # Skip invalid rows
                    
                    text = row[0].strip()
                    if not text:
                        continue
                    
                    if data_type == 'risk' or data_type == 'unknown':
                        # Try to process as risk data
                        risk_example = process_risk_csv_row(row, line_count)
                        if risk_example:
                            examples.append(risk_example)
                            valid_count += 1
                    
                    if data_type == 'pii' or data_type == 'unknown':
                        # Try to process as PII data
                        pii_example = process_pii_csv_row(row, line_count)
                        if pii_example:
                            examples.append(pii_example)
                            valid_count += 1
                    
                    if data_type == 'mixed':
                        # For mixed data, we need to determine the type for each row
                        second_col = row[1].strip() if len(row) > 1 else ""
                        if second_col.upper() in ('PC0', 'PC1', 'PC3'):
                            pii_example = process_pii_csv_row(row, line_count)
                            if pii_example:
                                examples.append(pii_example)
                                valid_count += 1
                        else:
                            risk_example = process_risk_csv_row(row, line_count)
                            if risk_example:
                                examples.append(risk_example)
                                valid_count += 1
                    
                    # Print progress periodically
                    if valid_count % 10000 == 0:
                        print(f"Processed {line_count} rows, found {valid_count} valid examples...")
                
                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        print(f"Error processing row {line_count}: {str(e)}")
        
        print(f"CSV processing complete. Total rows: {line_count}, Valid examples: {valid_count}, Errors: {errors}")
        
    except Exception as e:
        print(f"Error loading CSV training data: {str(e)}")
        traceback.print_exc()
    
    return examples

def process_risk_csv_row(row, line_count):
    """Process a CSV row for risk categorization task."""
    # Parse macro risk
    macro_risk_raw = row[1].strip()
    macro_risk_id = None
    
    # Check if it's a number or a full category name
    if macro_risk_raw.isdigit():
        macro_risk_id = macro_risk_raw
    else:
        # Extract number from format like "1. Operating Model & Risk Management"
        parts = macro_risk_raw.split('.')
        if parts[0].isdigit():
            macro_risk_id = parts[0]
        else:
            # Try to match by name
            for key, value in MACRO_RISKS.items():
                if value.lower() in macro_risk_raw.lower():
                    macro_risk_id = key
                    break
    
    if not macro_risk_id or macro_risk_id not in MACRO_RISKS:
        if line_count <= 5:  # Only show the first few errors
            print(f"Warning: Invalid macro risk category '{macro_risk_raw}' in row {line_count}")
        return None
    
    # Format the macro risk properly
    macro_risk = f"{macro_risk_id}. {MACRO_RISKS[macro_risk_id]}"
    
    # Parse thematic risks
    risk_themes = []
    valid_themes = set(THEMATIC_RISKS[macro_risk_id])
    
    # If we have a single column with comma-separated themes
    if len(row) == 3 and ',' in row[2]:
        themes_raw = [t.strip() for t in row[2].split(',')]
        for theme in themes_raw:
            if theme in valid_themes:
                risk_themes.append(theme)
    # If we have multiple columns for themes
    else:
        for i in range(2, len(row)):
            theme = row[i].strip()
            if theme and theme in valid_themes:
                risk_themes.append(theme)
    
    if not risk_themes:
        if line_count <= 5:
            print(f"Warning: No valid thematic risks found in row {line_count}")
        return None
    
    return {
        "type": "risk",
        "text": row[0].strip(),
        "macro_risk": macro_risk,
        "risk_themes": risk_themes
    }

def process_pii_csv_row(row, line_count):
    """Process a CSV row for PII classification task."""
    text = row[0].strip()
    pc_category = row[1].strip().upper()
    
    # Validate PC category
    if pc_category not in PII_PROTECTION_CATEGORIES:
        if line_count <= 5:
            print(f"Warning: Invalid PC category '{pc_category}' in row {line_count}. Must be PC0, PC1, or PC3.")
        return None
    
    # Parse PII types if provided
    pii_types = []
    if len(row) > 2:
        # If we have a single column with comma-separated PII types
        if len(row) == 3 and ',' in row[2]:
            pii_types = [t.strip() for t in row[2].split(',') if t.strip()]
        # If we have multiple columns for PII types
        else:
            for i in range(2, len(row)):
                pii_type = row[i].strip()
                if pii_type:
                    pii_types.append(pii_type)
    
    # PII types are optional for PC0 (no PII)
    if pc_category != 'PC0' and not pii_types:
        if line_count <= 5:
            print(f"Warning: No PII types found for {pc_category} in row {line_count}")
    
    return {
        "type": "pii",
        "text": text,
        "pc_category": pc_category,
        "pii_types": pii_types
    }

def load_json_training_data(file_path: str, data_type: str) -> List[Dict[str, Any]]:
    """
    Load training data from a JSON or JSONL file.
    
    Expected format for each example:
    {
        "risk_finding": "Text description of the risk",
        "macro_risk": "8. Manage IT Vulnerabilities & Patching" or "8" or "Manage IT Vulnerabilities & Patching",
        "risk_themes": ["Patching Completeness", "Vulnerability assessment and risk treatment"]
    }
    
    Args:
        file_path: Path to the JSON/JSONL file
        data_type: 'risk', 'pii', 'mixed', or 'unknown'
        
    Returns:
        List of training examples
    """
    examples = []
    processed = 0
    valid_count = 0
    errors = 0
    
    try:
        # Check if it's a JSONL file (one JSON object per line)
        if file_path.endswith('.jsonl'):
            print(f"Processing JSONL file: {file_path}")
            
            # Count total lines for progress reporting
            with open(file_path, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for _ in f)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(tqdm(f, total=total_lines, desc="Processing JSONL")):
                    processed += 1
                    try:
                        if not line.strip():
                            continue
                        
                        data = json.loads(line)
                        example = process_json_example(data, data_type)
                        if example:
                            examples.append(example)
                            valid_count += 1
                    except Exception as e:
                        errors += 1
                        if errors <= 5:
                            print(f"Error processing line {line_num+1}: {str(e)}")
                    
                    # Print progress periodically
                    if processed % 10000 == 0:
                        print(f"Processed {processed} items, found {valid_count} valid examples...")
        else:
            # Regular JSON file (single object or array)
            print(f"Processing JSON file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if isinstance(data, list):
                    print(f"Processing JSON array with {len(data)} items...")
                    for i, item in enumerate(tqdm(data, desc="Processing JSON array")):
                        processed += 1
                        try:
                            example = process_json_example(item, data_type)
                            if example:
                                examples.append(example)
                                valid_count += 1
                        except Exception as e:
                            errors += 1
                            if errors <= 5:
                                print(f"Error processing item {i+1}: {str(e)}")
                        
                        # Print progress periodically
                        if processed % 10000 == 0:
                            print(f"Processed {processed} items, found {valid_count} valid examples...")
                else:
                    print("Processing single JSON object...")
                    example = process_json_example(data, data_type)
                    if example:
                        examples.append(example)
                        valid_count += 1
        
        print(f"JSON processing complete. Processed: {processed}, Valid: {valid_count}, Errors: {errors}")
    
    except Exception as e:
        print(f"Error loading JSON training data: {str(e)}")
        traceback.print_exc()
    
    return examples

def process_json_example(data: Dict, data_type: str) -> Optional[Dict[str, Any]]:
    """
    Process a single JSON example and validate it against the standardized categories.
    
    Args:
        data: JSON data for a single example
        data_type: 'risk', 'pii', 'mixed', or 'unknown'
        
    Returns:
        Dictionary with processed example, or None if invalid
    """
    if not isinstance(data, dict):
        return None
    
    if 'risk_finding' not in data or not data['risk_finding']:
        return None
    
    risk_finding = data['risk_finding']
    
    # Parse macro risk
    if 'macro_risk' not in data or not data['macro_risk']:
        return None
    
    macro_risk_raw = data['macro_risk']
    macro_risk_id = None
    
    # Check different formats of macro_risk
    if isinstance(macro_risk_raw, str):
        if macro_risk_raw.isdigit():
            macro_risk_id = macro_risk_raw
        else:
            # Extract number from format like "1. Operating Model & Risk Management"
            parts = macro_risk_raw.split('.')
            if parts[0].isdigit():
                macro_risk_id = parts[0]
            else:
                # Try to match by name
                for key, value in MACRO_RISKS.items():
                    if value.lower() in macro_risk_raw.lower():
                        macro_risk_id = key
                        break
    elif isinstance(macro_risk_raw, int):
        macro_risk_id = str(macro_risk_raw)
    
    if not macro_risk_id or macro_risk_id not in MACRO_RISKS:
        return None
    
    # Format the macro risk properly
    macro_risk = f"{macro_risk_id}. {MACRO_RISKS[macro_risk_id]}"
    
    # Parse thematic risks
    if 'risk_themes' not in data or not isinstance(data['risk_themes'], list):
        return None
    
    risk_themes = []
    valid_themes = set(THEMATIC_RISKS[macro_risk_id])
    
    for theme in data['risk_themes']:
        if theme in valid_themes:
            risk_themes.append(theme)
    
    if not risk_themes:
        return None
    
    return {
        "type": data_type,
        "text": risk_finding,
        "macro_risk": macro_risk,
        "risk_themes": risk_themes
    }

def load_excel_training_data(file_path: str, data_type: str) -> List[Dict[str, Any]]:
    """
    Load training data from an Excel file.
    
    Expected format similar to CSV:
    - Column 1: Risk finding text
    - Column 2: Macro risk category
    - Column 3+: Thematic risks
    
    Args:
        file_path: Path to the Excel file
        data_type: 'risk', 'pii', 'mixed', or 'unknown'
        
    Returns:
        List of training examples
    """
    examples = []
    valid_count = 0
    errors = 0
    
    try:
        print(f"Loading Excel file {file_path}...")
        
        # Use pandas to efficiently read Excel files
        # We'll process it in chunks to handle large files
        chunks = pd.read_excel(file_path, chunksize=10000)
        chunk_num = 0
        
        for df_chunk in chunks:
            chunk_num += 1
            print(f"Processing chunk {chunk_num} with {len(df_chunk)} rows...")
            
            for index, row in tqdm(df_chunk.iterrows(), total=len(df_chunk), desc=f"Processing chunk {chunk_num}"):
                try:
                    if len(row) < 2:
                        continue  # Skip invalid rows
                    
                    risk_finding = str(row[0]).strip()
                    if not risk_finding or pd.isna(risk_finding):
                        continue
                    
                    # Parse macro risk
                    macro_risk_raw = str(row[1]).strip() if not pd.isna(row[1]) else ""
                    if not macro_risk_raw:
                        continue
                        
                    macro_risk_id = None
                    
                    # Check if it's a number or a full category name
                    if macro_risk_raw.isdigit():
                        macro_risk_id = macro_risk_raw
                    else:
                        # Extract number from format like "1. Operating Model & Risk Management"
                        parts = macro_risk_raw.split('.')
                        if parts[0].isdigit():
                            macro_risk_id = parts[0]
                        else:
                            # Try to match by name
                            for key, value in MACRO_RISKS.items():
                                if value.lower() in macro_risk_raw.lower():
                                    macro_risk_id = key
                                    break
                    
                    if not macro_risk_id or macro_risk_id not in MACRO_RISKS:
                        errors += 1
                        if errors <= 5:
                            print(f"Warning: Invalid macro risk category '{macro_risk_raw}' in row {index+2}")
                        continue
                    
                    # Format the macro risk properly
                    macro_risk = f"{macro_risk_id}. {MACRO_RISKS[macro_risk_id]}"
                    
                    # Parse thematic risks
                    risk_themes = []
                    valid_themes = set(THEMATIC_RISKS[macro_risk_id])
                    
                    # Check if we have a single column with comma-separated themes
                    if len(row) >= 3 and not pd.isna(row[2]) and ',' in str(row[2]):
                        themes_raw = str(row[2]).split(',')
                        for theme in themes_raw:
                            theme = theme.strip()
                            if theme in valid_themes:
                                risk_themes.append(theme)
                    # Or multiple columns for themes
                    else:
                        for i in range(2, len(row)):
                            if pd.notna(row[i]):
                                theme = str(row[i]).strip()
                                if theme and theme in valid_themes:
                                    risk_themes.append(theme)
                    
                    if not risk_themes:
                        errors += 1
                        if errors <= 5:
                            print(f"Warning: No valid thematic risks found in row {index+2}")
                        continue
                    
                    examples.append({
                        "type": data_type,
                        "text": risk_finding,
                        "macro_risk": macro_risk,
                        "risk_themes": risk_themes
                    })
                    valid_count += 1
                    
                    # Print progress periodically within the chunk
                    if valid_count % 10000 == 0:
                        print(f"Found {valid_count} valid examples so far...")
                
                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        print(f"Error processing row {index+2}: {str(e)}")
            
            # Force garbage collection between chunks
            gc.collect()
        
        print(f"Excel processing complete. Valid examples: {valid_count}, Errors: {errors}")
    
    except Exception as e:
        print(f"Error loading Excel training data: {str(e)}")
        traceback.print_exc()
    
    return examples

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
    
    # Load training data
    try:
        print("Loading and processing training data...")
        training_examples = []
        
        # Process input files
        if os.path.isdir(training_data_path):
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
        
        if risk_count == 0:
            print("Warning: No risk categorization examples found. Model will only learn PII detection.")
        
        if pii_count == 0:
            print("Warning: No PII detection examples found. Model will only learn risk categorization.")
        
        # Split data
        random.shuffle(training_examples)
        split_idx = int(len(training_examples) * 0.9)
        train_examples = training_examples[:split_idx]
        eval_examples = training_examples[split_idx:]
        
        print(f"Created {len(train_examples)} training examples and {len(eval_examples)} evaluation examples")
        
        # Process in batches if needed
        large_dataset = len(train_examples) > 10000
        batch_size = 5000 if large_dataset else len(train_examples)
        
        # Create training and evaluation files
        train_file = os.path.join(output_dir, "unified_train.jsonl")
        eval_file = os.path.join(output_dir, "unified_eval.jsonl")
        
        with open(train_file, 'w', encoding='utf-8') as f:
            if large_dataset:
                print(f"Processing large dataset in batches of {batch_size}...")
                for i in range(0, len(train_examples), batch_size):
                    batch = train_examples[i:i+batch_size]
                    process_batch(batch, f)
                    gc.collect()
            else:
                process_batch(train_examples, f)
        
        with open(eval_file, 'w', encoding='utf-8') as f:
            process_batch(eval_examples, f)
        
        # Now perform the actual fine-tuning
        print("\nStarting the fine-tuning process...")
        try:
            # Check if required packages are installed
            import importlib
            if importlib.util.find_spec("transformers") is None:
                print("Installing required packages for fine-tuning...")
                import subprocess
                subprocess.check_call(["pip", "install", "-q", "transformers", "datasets", "accelerate", "peft", "torch", "bitsandbytes"])
            
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
            
            # Check if CUDA is available
            if torch.cuda.is_available():
                print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
                device = "cuda"
                
                # Check VRAM
                vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                print(f"Available VRAM: {vram_mb:.2f} MB")
                
                # Adjust batch size based on VRAM
                if vram_mb < 16000:  # Less than 16GB
                    bs = 1
                    ga_steps = 8
                    print("Limited VRAM detected, using smaller batch size and more gradient accumulation steps")
                else:
                    bs = 2
                    ga_steps = 4
            else:
                print("CUDA is not available. Using CPU for training (this will be very slow).")
                print("Consider using a machine with a GPU for training.")
                device = "cpu"
                bs = 1
                ga_steps = 8
            
            # Configure training to use quantization and LoRA for memory efficiency
            print("Configuring model for efficient fine-tuning...")
            
            # Load the model with quantization
            model = AutoModelForCausalLM.from_pretrained(
                "fdtn-ai/Foundation-Sec-8B",
                load_in_8bit=True,
                device_map="auto",
                torch_dtype=torch.float16
            )
            model = prepare_model_for_kbit_training(model)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained("fdtn-ai/Foundation-Sec-8B")
            tokenizer.pad_token = tokenizer.eos_token
            
            # Configure LoRA
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
            
            # Load dataset
            train_dataset = load_dataset('json', data_files=train_file, split='train')
            eval_dataset = load_dataset('json', data_files=eval_file, split='train')
            
            # Get number of training steps
            num_epochs = 3
            total_steps = len(train_dataset) * num_epochs // (bs * ga_steps)
            eval_steps = max(100, total_steps // 10)  # Evaluate roughly 10 times during training
            save_steps = max(200, total_steps // 5)   # Save roughly 5 checkpoints during training
            
            print(f"Training for {num_epochs} epochs with {total_steps} total steps")
            print(f"Evaluating every {eval_steps} steps, saving every {save_steps} steps")
            
            # Define training arguments
            training_args = TrainingArguments(
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
                evaluation_strategy="steps",
                save_strategy="steps",
                load_best_model_at_end=True,
                fp16=True,
                report_to="none",
                save_total_limit=3,  # Keep only the 3 best checkpoints to save disk space
                logging_first_step=True,
                dataloader_num_workers=4,
                dataloader_drop_last=True
            )
            
            # Prepare the data
            def format_chat(example):
                return {"input_ids": tokenizer.encode(tokenizer.apply_chat_template(example["messages"], tokenize=False))}
            
            print("Processing training dataset...")
            train_dataset = train_dataset.map(format_chat, remove_columns=["messages"])
            
            print("Processing evaluation dataset...")
            eval_dataset = eval_dataset.map(format_chat, remove_columns=["messages"])
            
            # Initialize Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            )
            
            # Train the model
            print("Starting training...")
            trainer.train()
            
            # Save the final model
            print("Training complete. Saving the model...")
            final_model_path = os.path.join(output_dir, "final_model")
            trainer.save_model(final_model_path)
            
            # Save the tokenizer
            tokenizer.save_pretrained(final_model_path)
            
            # Free up memory before creating the pickle file
            del model
            del trainer
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Create a pickle file for easier inference
            pickle_path = os.path.join(output_dir, "unified_model_with_categories.pkl")
            print(f"Creating pickle file for inference at: {pickle_path}")
            
            # Create a dictionary with the model parameters and all categories
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
            
            print(f"\nFine-tuning complete!")
            print(f"Trained unified model saved to: {final_model_path}")
            print(f"Inference package saved to: {pickle_path}")
            
            return pickle_path
            
        except Exception as e:
            print(f"\nError during fine-tuning: {str(e)}")
            traceback.print_exc()
            
            print("\nFine-tuning failed. Training data has been prepared, and you can run fine-tuning manually with the following command:")
            script_file = os.path.join(output_dir, "run_fine_tuning.sh")
            with open(script_file, 'w') as f:
                f.write(f"""#!/bin/bash
# Fine-tuning script for Foundation-Sec-8B model

# Install required packages
pip install -q transformers datasets accelerate peft bitsandbytes torch

# Run the fine-tuning
python -m transformers.trainer \\
  --model_name_or_path="fdtn-ai/Foundation-Sec-8B" \\
  --train_file="{train_file}" \\
  --validation_file="{eval_file}" \\
  --per_device_train_batch_size=2 \\
  --per_device_eval_batch_size=2 \\
  --gradient_accumulation_steps=4 \\
  --num_train_epochs=3 \\
  --learning_rate=2e-5 \\
  --weight_decay=0.01 \\
  --warmup_ratio=0.03 \\
  --lr_scheduler_type="cosine" \\
  --logging_steps=10 \\
  --evaluation_strategy="steps" \\
  --save_strategy="steps" \\
  --save_steps=200 \\
  --output_dir="./finetuned-foundation-sec-8b" \\
  --bf16 \\
  --overwrite_output_dir
""")
            
            os.chmod(script_file, 0o755)  # Make the script executable
            print(f"Manual fine-tuning script created: {script_file}")
            return None
    
    except Exception as e:
        print(f"Error processing training data: {str(e)}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune the Foundation-Sec-8B model for both risk categorization and PII detection")
    parser.add_argument("--training_data", type=str, required=True, help="Path to training data file or directory")
    parser.add_argument("--output_dir", type=str, default="fine_tuning_data", help="Directory to save fine-tuning output")
    
    args = parser.parse_args()
    
    pickle_path = fine_tune_model(args.training_data, args.output_dir)
    
    if pickle_path:
        print(f"Fine-tuning complete. Unified model saved to {args.output_dir}/final_model")
        print(f"Inference package saved to {pickle_path}")
        print("\nTo use the model for inference, run:")
        print(f"python risk_inference.py --model {pickle_path} --text \"Your text to analyze\"")
    else:
        print("Fine-tuning failed.") 