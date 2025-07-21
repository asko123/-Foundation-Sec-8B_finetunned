#!/usr/bin/env python3
"""
Enhanced Risk & PII Fine-Tuner - A tool for fine-tuning the Foundation-Sec-8B model.

This enhanced version can process raw Excel/CSV files from a folder and automatically
extract security risk findings and PII data to create training examples.

Key Features:
1. Automatic folder scanning and file processing
2. Raw data extraction from Excel/CSV files
3. Intelligent content analysis and categorization
4. Knowledge extraction from unstructured data
5. Training example generation from inferred categories
"""

# Standard library imports
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
from rapidfuzz import fuzz, process, utils
from collections import Counter
import numpy as np

# Import constants
from risk_fine_tuner_constants import (
    L2,
    MACRO_RISKS,
    PII_PROTECTION_CATEGORIES,
    PII_TYPES,
    PRIVACY_CLASSIFICATIONS,
    SENSITIVITY_LEVELS,
    RISK_KEYWORDS,
    PII_KEYWORDS
)

def scan_folder_for_files(folder_path: str) -> List[str]:
    """Scan folder for Excel and CSV files."""
    supported_extensions = ['.xlsx', '.xls', '.csv']
    files_to_process = []
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in supported_extensions):
                files_to_process.append(os.path.join(root, file))
    
    return files_to_process

def calculate_string_similarity(str1: str, str2: str, score_cutoff: float = 65) -> float:
    """
    Calculate string similarity using rapidfuzz's advanced features.
    
    Args:
        str1: First string to compare
        str2: Second string to compare
        score_cutoff: Minimum score threshold (0-100)
        
    Returns:
        Similarity score between 0 and 1
    """
    # Preprocess strings
    str1_proc = utils.default_process(str1)
    str2_proc = utils.default_process(str2)
    
    if not str1_proc or not str2_proc:
        return 0.0
    
    # Exact match
    if str1_proc == str2_proc:
        return 1.0
    
    # Calculate different types of ratios
    ratios = [
        fuzz.ratio(str1_proc, str2_proc, score_cutoff=score_cutoff),
        fuzz.partial_ratio(str1_proc, str2_proc, score_cutoff=score_cutoff),
        fuzz.token_sort_ratio(str1_proc, str2_proc, score_cutoff=score_cutoff),
        fuzz.token_set_ratio(str1_proc, str2_proc, score_cutoff=score_cutoff),
        fuzz.WRatio(str1_proc, str2_proc, score_cutoff=score_cutoff)
    ]
    
    # Return the highest score normalized to 0-1 range
    return max((score for score in ratios if score is not None), default=0) / 100.0

def find_best_matches(query: str, choices: List[str], limit: int = 3, score_cutoff: float = 0.65) -> List[Tuple[str, float]]:
    """
    Find the best matching strings from a list of choices using rapidfuzz's process.
    
    Args:
        query: String to match against
        choices: List of strings to search through
        limit: Maximum number of matches to return
        score_cutoff: Minimum similarity score (0-1)
        
    Returns:
        List of tuples (matched_string, score)
    """
    # Use rapidfuzz's process.extract for efficient matching
    matches = process.extract(
        query,
        choices,
        scorer=fuzz.WRatio,
        score_cutoff=score_cutoff * 100,
        limit=limit,
        processor=utils.default_process
    )
    
    # Convert scores to 0-1 range
    return [(match[0], match[1] / 100.0) for match in matches]

def analyze_content_for_risk_category(text: str) -> Tuple[str, List[str], float]:
    """
    Analyze text content to identify L2 category and macro risks using improved matching.
    
    Args:
        text: Text content to analyze
        
    Returns:
        Tuple of (l2_category, macro_risks, confidence_score)
    """
    # Initialize variables
    best_l2 = None
    best_l2_score = 0
    identified_risks = set()
    
    # Find best L2 category match
    l2_matches = process.extract(
        text,
        {key: value for key, value in L2.items()}.values(),
        scorer=fuzz.WRatio,
        score_cutoff=75,  # Require higher confidence for L2 categories
        limit=1,
        processor=utils.default_process
    )
    
    if l2_matches:
        matched_value = l2_matches[0][0]
        best_l2_score = l2_matches[0][1] / 100.0
        # Find the key for this value
        for key, value in L2.items():
            if value == matched_value:
                best_l2 = f"{key}. {value}"
                break
    
    # If we found an L2 category, look for associated macro risks
    if best_l2:
        l2_key = best_l2.split('.')[0]
        available_risks = MACRO_RISKS.get(l2_key, [])
        
        # Extract potential risk phrases from text
        text_proc = utils.default_process(text)
        if text_proc:
            # Find all potential matches using token set ratio
            risk_matches = process.extract(
                text_proc,
                available_risks,
                scorer=fuzz.token_set_ratio,
                score_cutoff=65,  # More lenient for risks
                limit=None,  # Get all matches above score_cutoff
                processor=utils.default_process
            )
            
            # Add matched risks with their confidence scores
            for risk, score, _ in risk_matches:
                identified_risks.add(risk)
                print(f"Matched risk: '{risk}' (confidence: {score/100:.2f})")
    
    return best_l2, list(identified_risks), best_l2_score

def analyze_content_for_pii(text: str) -> Tuple[str, List[str], float]:
    """
    Analyze text content to identify PII and classify protection category.
    
    Args:
        text: Text content to analyze
        
    Returns:
        Tuple of (pc_category, pii_types, confidence_score)
    """
    text_lower = text.lower()
    
    # Initialize variables
    identified_pii = set()
    confidence_score = 0
    
    # Check for PII types
    for pii_type in PII_TYPES:
        type_lower = pii_type.lower()
        if type_lower in text_lower:
            identified_pii.add(pii_type)
            confidence_score += 0.2  # Increase confidence for each PII type found
    
    # Check for specific patterns
    patterns = {
        r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b': 'SSN',  # SSN pattern
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b': 'Email',  # Email
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b': 'Phone',  # Phone
        r'\b\d{1,5}\s+[A-Za-z\s]{2,30}\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard)\b': 'Address',
        r'\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b': 'DOB'  # Date pattern
    }
    
    for pattern, pii_type in patterns.items():
        if re.search(pattern, text):
            identified_pii.add(pii_type)
            confidence_score += 0.3  # Higher confidence for pattern matches
    
    # Determine privacy classification
    if any(pii in identified_pii for pii in ['SSN', 'Financial', 'Health', 'DOB']):
        pc_category = 'PC3'
        confidence_score = min(1.0, confidence_score + 0.4)
    elif identified_pii:
        pc_category = 'PC1'
        confidence_score = min(1.0, confidence_score + 0.2)
    else:
        pc_category = 'PC0'
        confidence_score = max(0.1, confidence_score)
    
    return pc_category, list(identified_pii), confidence_score

def process_raw_data_to_training_examples(raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process raw data entries into training examples."""
    training_examples = []
    debug_stats = {
        'total_entries': len(raw_data),
        'empty_entries': 0,
        'processed_entries': 0,
        'failed_entries': 0,
        'field_stats': {
            'title': 0,
            'description': 0,
            'l2': 0,
            'risks': 0
        }
    }
    
    print(f"Processing {len(raw_data)} entries...")
    
    for idx, entry in enumerate(raw_data):
        try:
            # Skip completely empty entries
            if not entry:
                debug_stats['empty_entries'] += 1
                continue
            
            # Create basic example structure
            example = {
                "type": "risk",
                "text": "",
                "l2_category": "UNKNOWN",
                "macro_risks": ["UNSPECIFIED"],
                "metadata": {"source": "raw_data", "original_entry": entry}
            }
            
            # Build text content from any available fields
            text_parts = []
            
            # Try all possible title fields
            title_fields = ['Finding_Title', 'Title', 'FINDING_TITLE', 'TITLE', 'Name', 'NAME', 'Summary', 'SUMMARY']
            for field in title_fields:
                if entry.get(field) and is_meaningful_text(entry[field]):
                    text_parts.append(f"Finding: {clean_text(entry[field])}")
                    debug_stats['field_stats']['title'] += 1
                    break
            
            # Try all possible description fields
            desc_fields = ['Finding_Description', 'Description', 'FINDING_DESCRIPTION', 'DESCRIPTION', 
                         'Details', 'DETAILS', 'Content', 'CONTENT']
            for field in desc_fields:
                if entry.get(field) and is_meaningful_text(entry[field]):
                    text_parts.append(f"Description: {clean_text(entry[field])}")
                    debug_stats['field_stats']['description'] += 1
                    break
            
            # Look for any other text fields
            for key, value in entry.items():
                if (isinstance(value, str) and 
                    is_meaningful_text(value) and 
                    key not in title_fields + desc_fields):
                    text_parts.append(f"{key}: {clean_text(value)}")
            
            # Try to extract L2 category
            l2_fields = ['L2', 'Category', 'Type', 'Classification', 'Risk_Category']
            for field in l2_fields:
                if entry.get(field):
                    example['l2_category'] = str(entry[field])
                    debug_stats['field_stats']['l2'] += 1
                    break
            
            # Try to extract macro risks
            risk_fields = ['macro_risks', 'Risks', 'Risk_Types', 'Categories']
            for field in risk_fields:
                if entry.get(field):
                    risks = entry[field]
                    if isinstance(risks, list):
                        example['macro_risks'] = [str(r) for r in risks]
                    elif isinstance(risks, str):
                        # Try various delimiters
                        for delimiter in [',', ';', '|', '\n']:
                            if delimiter in risks:
                                example['macro_risks'] = [r.strip() for r in risks.split(delimiter)]
                                break
                        if example['macro_risks'] == ["UNSPECIFIED"]:
                            example['macro_risks'] = [risks.strip()]
                    debug_stats['field_stats']['risks'] += 1
                    break
            
            # Accept example if it has any meaningful content
            if text_parts:
                example['text'] = "\n\n".join(text_parts)
                training_examples.append(example)
                debug_stats['processed_entries'] += 1
            else:
                debug_stats['failed_entries'] += 1
        
        except Exception as e:
            print(f"Error processing entry {idx}: {str(e)}")
            debug_stats['failed_entries'] += 1
            traceback.print_exc()
    
    # Print only essential statistics
    print(f"\nProcessed {debug_stats['total_entries']} entries:")
    print(f"- Successfully extracted: {debug_stats['processed_entries']} ({debug_stats['processed_entries']/debug_stats['total_entries']*100:.1f}%)")
    if debug_stats['failed_entries'] > 0:
        print(f"- Failed to process: {debug_stats['failed_entries']} ({debug_stats['failed_entries']/debug_stats['total_entries']*100:.1f}%)")
    
    return training_examples

def process_folder_for_training_data(folder_path: str, output_dir: str = "training_data") -> str:
    """Process all files in a folder to extract training data."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Scan folder for files
    files_to_process = scan_folder_for_files(folder_path)
    
    if not files_to_process:
        raise ValueError(f"No supported files found in folder: {folder_path}")
    
    # Extract raw data from all files
    all_raw_data = []
    
    for file_path in files_to_process:
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in ['.xlsx', '.xls']:
            raw_data = extract_text_from_excel(file_path)
        elif file_ext == '.csv':
            raw_data = extract_text_from_csv(file_path)
        else:
            print(f"Skipping unsupported file type: {file_ext}")
            continue
        
        if raw_data:
            all_raw_data.extend(raw_data)
    
    if not all_raw_data:
        raise ValueError("No raw data could be extracted from any files")
    
    # Convert raw data to training examples
    training_examples = process_raw_data_to_training_examples(all_raw_data)
    
    if not training_examples:
        # Create a backup file with the raw data for debugging
        backup_file = os.path.join(output_dir, "raw_data_backup.jsonl")
        with open(backup_file, 'w', encoding='utf-8') as f:
            for entry in all_raw_data:
                f.write(json.dumps(entry) + '\n')
        print(f"\nERROR: No training examples could be generated.")
        print(f"Raw data saved to: {backup_file}")
        print("Common issues:")
        print("1. Missing or invalid L2 categories")
        print("2. Missing or invalid macro risks")
        print("3. Empty or invalid text content")
        raise ValueError("Could not generate any training examples, even with relaxed requirements")
    
    # Save training examples to JSONL file
    training_file_path = os.path.join(output_dir, "auto_generated_training_data.jsonl")
    
    with open(training_file_path, 'w', encoding='utf-8') as f:
        for example in training_examples:
            f.write(json.dumps(example) + '\n')
    
    # Save summary report
    summary_path = os.path.join(output_dir, "extraction_summary.json")
    summary = {
        "total_files_processed": len(files_to_process),
        "total_raw_entries": len(all_raw_data),
        "total_training_examples": len(training_examples),
        "risk_examples": len([ex for ex in training_examples if ex["type"] == "risk"]),
        "pii_examples": len([ex for ex in training_examples if ex["type"] == "pii"]),
        "example_stats": {
            "with_l2": len([ex for ex in training_examples if ex["l2_category"] != "UNKNOWN"]),
            "with_risks": len([ex for ex in training_examples if ex["macro_risks"] != ["UNSPECIFIED"]]),
            "complete": len([ex for ex in training_examples 
                           if ex["l2_category"] != "UNKNOWN" and ex["macro_risks"] != ["UNSPECIFIED"]])
        }
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nExtracted {len(training_examples)} examples:")
    print(f"- Complete examples: {summary['example_stats']['complete']}")
    print(f"- With L2 only: {summary['example_stats']['with_l2'] - summary['example_stats']['complete']}")
    print(f"- With risks only: {summary['example_stats']['with_risks'] - summary['example_stats']['complete']}")
    print(f"- Partial examples: {len(training_examples) - summary['example_stats']['complete']}")
    
    return training_file_path

def format_all_categories_for_prompt() -> str:
    """Format all categories for inclusion in prompts."""
    categories_text = "PART 1: SECURITY RISK CATEGORIES\n\n"
    categories_text += "L2 Categories:\n"
    for key, value in L2.items():
        categories_text += f"{key}. {value}\n"
    
    categories_text += "\nMacro Risks for each L2 Category:\n"
    for key, risks in MACRO_RISKS.items():
        categories_text += f"\n{key}. {L2[key]}:\n"
        for risk in risks:
            categories_text += f"   - {risk}\n"
    
    categories_text += "\n\nPART 2: PII PROTECTION CATEGORIES\n\n"
    categories_text += "PII Protection Categories:\n"
    for key, value in PII_PROTECTION_CATEGORIES.items():
        categories_text += f"{key}: {value}\n"
    
    categories_text += "\nCommon PII Types:\n"
    for pii_type in PII_TYPES:
        categories_text += f"- {pii_type}\n"
            
    return categories_text

# Continue with the rest of the functions from the original file...
# [The remaining functions would be the same as in the original file, starting from load_training_file onwards]

def load_training_file(file_path: str) -> List[Dict[str, Any]]:
    """Load training data from JSONL file generated by this enhanced processor."""
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
        print(f"Error loading training file {file_path}: {e}")
    
    return examples

def analyze_skipped_entries(raw_data: pd.DataFrame, skipped_reasons: Dict[str, int]) -> Dict[str, Any]:
    """Analyze skipped entries to understand common patterns."""
    analysis = {
        'l2_patterns': {},
        'risk_patterns': {},
        'empty_fields': {
            'title_only': 0,
            'description_only': 0,
            'both_empty': 0
        },
        'common_values': {
            'l2': Counter(),
            'risks': Counter()
        }
    }
    
    for _, row in raw_data.iterrows():
        # Analyze empty fields
        if pd.isna(row['FINDING_TITLE']) and pd.isna(row['FINDING_DESCRIPTION']):
            analysis['empty_fields']['both_empty'] += 1
        elif pd.isna(row['FINDING_TITLE']):
            analysis['empty_fields']['title_only'] += 1
        elif pd.isna(row['FINDING_DESCRIPTION']):
            analysis['empty_fields']['description_only'] += 1
            
        # Analyze L2 patterns
        if pd.notna(row['L2']):
            l2_value = str(row['L2']).strip()
            analysis['common_values']['l2'][l2_value] += 1
            
        # Analyze risk patterns
        if pd.notna(row['MACRO_RISKS']):
            risks = row['MACRO_RISKS']
            if isinstance(risks, str):
                analysis['common_values']['risks'][risks] += 1
            elif isinstance(risks, (list, tuple)):
                for risk in risks:
                    analysis['common_values']['risks'][str(risk)] += 1
    
    return analysis

def is_meaningful_text(text: str) -> bool:
    """Check if text contains meaningful content."""
    if pd.isna(text):
        return False
    text = str(text).strip()
    # Check if text is empty or just special characters/whitespace
    if not text or re.match(r'^[\s\W]*$', text):
        return False
    # Check if text is too short or just numbers
    if len(text) < 3 or text.isdigit():
        return False
    return True

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if pd.isna(text):
        return ""
    # Convert to string and strip whitespace
    text = str(text).strip()
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters that don't add meaning
    text = re.sub(r'[^\w\s\-.,;:?!()]', ' ', text)
    # Remove empty brackets
    text = re.sub(r'\(\s*\)', '', text)
    return text.strip()

def process_findings_data(raw_data: pd.DataFrame) -> List[Dict[str, Any]]:
    """Process raw findings data into training examples with improved matching."""
    training_examples = []
    skipped_reasons = {
        'no_l2': 0,
        'no_risks': 0,
        'empty_data': 0,
        'error': 0,
        'invalid_format': 0
    }
    
    try:
        # Clean column names and handle variations
        raw_data.columns = [str(col).strip().upper() for col in raw_data.columns]
        
        # Add more column variations
        column_mapping = {
            'FINDING_TITLE': [
                'FINDING_TITLE', 'TITLE', 'FINDING NAME', 'NAME', 'FINDING', 'SUMMARY', 'ISSUE_TITLE', 'ISSUE',
                'OBSERVATION_TITLE', 'VULNERABILITY_TITLE', 'VULN_TITLE', 'HEADING', 'SUBJECT'
            ],
            'FINDING_DESCRIPTION': [
                'FINDING_DESCRIPTION', 'DESCRIPTION', 'DESC', 'DETAILS', 'FINDING DETAILS', 'ISSUE_DESCRIPTION',
                'DETAIL', 'OBSERVATION', 'OBSERVATION_DETAILS', 'VULNERABILITY_DESCRIPTION', 'VULN_DESCRIPTION',
                'FINDING_DETAILS', 'NOTES', 'COMMENTS', 'CONTENT', 'BODY'
            ],
            'L2': [
                'L2', 'L2_CATEGORY', 'CATEGORY', 'RISK_CATEGORY', 'FINDING_CATEGORY', 'CLASSIFICATION', 'TYPE',
                'FINDING_TYPE', 'VULNERABILITY_TYPE', 'VULN_TYPE', 'RISK_TYPE', 'ISSUE_TYPE', 'SEVERITY',
                'RISK_CLASSIFICATION', 'SECURITY_CLASSIFICATION'
            ],
            'MACRO_RISKS': [
                'MACRO_RISKS', 'RISKS', 'RISK_TYPES', 'RISK_CATEGORIES', 'MACRO_RISK', 'RISK_TAGS', 'TAGS',
                'RISK_CLASSIFICATION', 'THREAT_TYPES', 'THREAT_CATEGORIES', 'IMPACT_TYPES', 'VULNERABILITY_TAGS',
                'SECURITY_TAGS', 'RISK_AREAS', 'RISK_DOMAINS'
            ]
        }
        
        # Try to find additional text columns that might contain useful information
        additional_text_columns = []
        for col in raw_data.columns:
            if col not in [item for sublist in column_mapping.values() for item in sublist]:
                # Sample some values from the column
                sample_values = raw_data[col].dropna().head(10)
                # Check if column contains text data
                if all(isinstance(val, str) and len(str(val)) > 10 for val in sample_values):
                    additional_text_columns.append(col)
        
        if additional_text_columns:
            print(f"\nFound additional text columns that might contain useful information: {additional_text_columns}")
        
        # Map columns to standard names
        mapped_columns = set()
        for standard_col, variations in column_mapping.items():
            if standard_col not in raw_data.columns:
                for var in variations:
                    if var in raw_data.columns:
                        raw_data = raw_data.rename(columns={var: standard_col})
                        mapped_columns.add(standard_col)
                        print(f"Mapped column '{var}' to '{standard_col}'")
                        break
        
        # Report unmapped columns
        unmapped = set(column_mapping.keys()) - mapped_columns
        if unmapped:
            print("\nWarning: Could not find matches for these columns:")
            for col in unmapped:
                print(f"- {col} (tried: {', '.join(column_mapping[col])})")
            print(f"Available columns: {', '.join(raw_data.columns)}")
        
        # Clean and normalize text data
        for col in ['FINDING_TITLE', 'FINDING_DESCRIPTION']:
            if col in raw_data.columns:
                raw_data[col] = raw_data[col].apply(clean_text)
        
        # Get list of additional columns for metadata
        metadata_columns = [col for col in raw_data.columns if col not in set(column_mapping.keys())]
        print(f"Found additional columns that will be preserved in metadata: {metadata_columns}")
        
        total_rows = len(raw_data)
        print(f"\nProcessing {total_rows} findings...")
        
        # Process in batches for better memory management
        batch_size = 5000
        for start_idx in range(0, total_rows, batch_size):
            end_idx = min(start_idx + batch_size, total_rows)
            batch = raw_data.iloc[start_idx:end_idx]
            
            print(f"\nProcessing batch {start_idx//batch_size + 1}/{(total_rows + batch_size - 1)//batch_size}")
            
            for idx, row in batch.iterrows():
                try:
                    if idx % 1000 == 0:
                        print(f"Progress: {idx}/{total_rows} rows processed")
                    
                    # Build context from all available text fields
                    context_parts = []
                    
                    # Add primary fields if they contain meaningful text
                    if 'FINDING_TITLE' in row and is_meaningful_text(row['FINDING_TITLE']):
                        context_parts.append(f"Finding: {row['FINDING_TITLE']}")
                    
                    if 'FINDING_DESCRIPTION' in row and is_meaningful_text(row['FINDING_DESCRIPTION']):
                        context_parts.append(f"Description: {row['FINDING_DESCRIPTION']}")
                    
                    # If primary fields are empty, try additional text columns
                    if not context_parts and additional_text_columns:
                        for col in additional_text_columns:
                            if is_meaningful_text(row[col]):
                                context_parts.append(f"{col}: {clean_text(row[col])}")
                    
                    # Only skip if we have no usable text at all
                    if not context_parts:
                        skipped_reasons['empty_data'] += 1
                        continue
                    
                    context = "\n\n".join(context_parts)
                    
                    # Parse L2 category with improved matching
                    l2_category = None
                    if pd.notna(row['L2']):
                        l2_value = str(row['L2']).strip().lower()
                        clean_l2_value = re.sub(r'[^\w\s]', '', l2_value)
                        
                        # Try exact matches first
                        if l2_value in l2_variations:
                            key, value = l2_variations[l2_value]
                            l2_category = f"{key}. {value}"
                        elif clean_l2_value in l2_variations:
                            key, value = l2_variations[clean_l2_value]
                            l2_category = f"{key}. {value}"
                        else:
                            # Try matching individual words
                            l2_words = set(clean_l2_value.split())
                            for word in l2_words:
                                if word in l2_variations:
                                    key, value = l2_variations[word]
                                    l2_category = f"{key}. {value}"
                                    print(f"Word matched L2: '{l2_value}' -> '{value}' (matched word: {word})")
                                    break
                            
                            # If still no match, use fuzzy matching
                            if not l2_category:
                                matches = process.extract(
                                    l2_value,
                                    list(L2.values()),
                                    scorer=fuzz.WRatio,
                                    score_cutoff=55,  # Even lower threshold
                                    limit=1,
                                    processor=utils.default_process
                                )
                                if matches:
                                    matched_value = matches[0][0]
                                    for key, value in L2.items():
                                        if value == matched_value:
                                            l2_category = f"{key}. {value}"
                                            print(f"Fuzzy matched L2: '{l2_value}' -> '{value}' (score: {matches[0][1]})")
                            break
                    
                    # Parse macro risks with improved handling
                    macro_risks = set()
                    if pd.notna(row['MACRO_RISKS']):
                        risks = row['MACRO_RISKS']
                        # Handle different input formats
                        if isinstance(risks, str):
                            # Try various delimiters
                            delimiters = [',', ';', '\n', '|', '/', '\\', '+', '&', 'and']
                            for delimiter in delimiters:
                                if delimiter in risks.lower():
                                    risks = [r.strip() for r in risks.split(delimiter)]
                                    break
                            if isinstance(risks, str):  # If no delimiter was found
                                risks = [risks.strip()]
                        elif isinstance(risks, (list, tuple)):
                            risks = [str(r).strip() for r in risks]
                        else:
                            skipped_reasons['invalid_format'] += 1
                            continue
                        
                        # Process each risk
                        for risk in risks:
                            if not risk:  # Skip empty strings
                                continue
                            
                            risk_lower = risk.lower()
                            matched = False
                            
                            # Try exact match first
                            for l2_key, risk_list in MACRO_RISKS.items():
                                if any(r.lower() == risk_lower for r in risk_list):
                                    matched_risk = next(r for r in risk_list if r.lower() == risk_lower)
                                    macro_risks.add(matched_risk)
                                    matched = True
                                    break
                            
                            # If no exact match, try fuzzy matching
                            if not matched:
                                # Get all possible risks for better matching
                                all_risks = [risk for risks in MACRO_RISKS.values() for risk in risks]
                                matches = process.extract(
                                    risk,
                                    all_risks,
                                    scorer=fuzz.WRatio,
                                    score_cutoff=55,  # Lower threshold
                                    limit=2,  # Try top 2 matches
                                    processor=utils.default_process
                                )
                                for match, score, _ in matches:
                                    macro_risks.add(match)
                                    print(f"Fuzzy matched risk: '{risk}' -> '{match}' (score: {score})")
                    
                    # If we have an L2 category but no risks, try to infer risks from the context
                    if l2_category and not macro_risks:
                        l2_key = l2_category.split('.')[0]
                        available_risks = MACRO_RISKS.get(l2_key, [])
                        
                        # Look for risk keywords in the context
                        context_lower = context.lower()
                        context_words = set(context_lower.split())
                        
                        for risk in available_risks:
                            risk_lower = risk.lower()
                            risk_words = set(risk_lower.split())
                            
                            # Check for word overlap or substring match
                            if (len(risk_words & context_words) >= min(2, len(risk_words)) or
                                any(word in context_lower for word in risk_words if len(word) > 4)):
                                macro_risks.add(risk)
                                print(f"Inferred risk from context: '{risk}'")
                    
                    # If we have risks but no L2, try to infer L2 from risks
                    if not l2_category and macro_risks:
                        risk_l2_counts = Counter()
                        for risk in macro_risks:
                            for l2_key, risks in MACRO_RISKS.items():
                                if risk in risks:
                                    risk_l2_counts[l2_key] += 1
                        
                        if risk_l2_counts:
                            most_likely_l2 = risk_l2_counts.most_common(1)[0][0]
                            l2_category = f"{most_likely_l2}. {L2[most_likely_l2]}"
                            print(f"Inferred L2 from risks: '{l2_category}'")
                    
                    # Collect metadata
                    metadata = {
                        "source": "raw_data",
                        "original_l2": row['L2'] if pd.notna(row['L2']) else None,
                        "original_risks": row['MACRO_RISKS'] if pd.notna(row['MACRO_RISKS']) else None,
                    }
                    
                    # Add additional metadata
                    for col in metadata_columns:
                        if pd.notna(row[col]):
                            metadata[col.lower()] = row[col]
                    
                    # Create example if we have either L2 or macro risks
                    if l2_category or macro_risks:
                        example = {
                            "type": "risk",
                            "text": context,
                            "l2_category": l2_category if l2_category else "UNKNOWN",
                            "macro_risks": list(macro_risks) if macro_risks else ["UNSPECIFIED"],
                            "metadata": metadata
                        }
                        training_examples.append(example)
                    else:
                        if not l2_category:
                            skipped_reasons['no_l2'] += 1
                        if not macro_risks:
                            skipped_reasons['no_risks'] += 1
                
                except Exception as e:
                    print(f"Error processing row {idx}: {str(e)}")
                    skipped_reasons['error'] += 1
                    continue
        
        # Analyze skipped entries
        print("\nAnalyzing skipped entries...")
        analysis = analyze_skipped_entries(raw_data, skipped_reasons)
        
        # Print detailed summary
        print(f"\nProcessed {total_rows} findings:")
        print(f"- Successfully extracted: {len(training_examples)} ({len(training_examples)/total_rows*100:.1f}%)")
        print("\nSkipped entries breakdown:")
        for reason, count in skipped_reasons.items():
            print(f"- {reason}: {count} ({count/total_rows*100:.1f}%)")
        
        print("\nEmpty fields analysis:")
        for field, count in analysis['empty_fields'].items():
            print(f"- {field}: {count} ({count/total_rows*100:.1f}%)")
        
        print("\nTop 10 unmatched L2 values:")
        for value, count in analysis['common_values']['l2'].most_common(10):
            print(f"- '{value}': {count} occurrences")
        
        print("\nTop 10 unmatched risk values:")
        for value, count in analysis['common_values']['risks'].most_common(10):
            print(f"- '{value}': {count} occurrences")
        
        if training_examples:
            # Print mapping statistics
            l2_stats = {}
            risk_stats = {}
            for ex in training_examples:
                if ex['l2_category'] != "UNKNOWN":
                    l2 = ex['l2_category'].split('.')[0]
                    l2_stats[l2] = l2_stats.get(l2, 0) + 1
                for risk in ex['macro_risks']:
                    if risk != "UNSPECIFIED":
                        risk_stats[risk] = risk_stats.get(risk, 0) + 1
            
            print("\nL2 Category Distribution:")
            for l2, count in sorted(l2_stats.items()):
                print(f"  L2 {l2}: {count} findings ({count/len(training_examples)*100:.1f}%)")
            
            print("\nTop Macro Risks:")
            for risk, count in sorted(risk_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {risk}: {count} occurrences ({count/len(training_examples)*100:.1f}%)")
        
    except Exception as e:
        print(f"Error processing findings data: {str(e)}")
        traceback.print_exc()
    
    return training_examples

def extract_text_from_excel(file_path: str) -> List[Dict[str, Any]]:
    """Extract text from Excel files."""
    try:
        print(f"Reading Excel file: {file_path}")
        
        # Try to read all sheets
        all_sheets = pd.read_excel(file_path, sheet_name=None)
        all_data = []
        
        for sheet_name, df in all_sheets.items():
            # Convert DataFrame to list of dictionaries
            sheet_data = df.replace({np.nan: None}).to_dict('records')
            
            # Add sheet name to metadata
            for record in sheet_data:
                record['_sheet_name'] = sheet_name
            
            all_data.extend(sheet_data)
        
        print(f"Extracted {len(all_data)} records from Excel file")
        return all_data
            
    except Exception as e:
        print(f"Error processing Excel file {file_path}: {str(e)}")
        traceback.print_exc()
        return []

def extract_text_from_csv(file_path: str) -> List[Dict[str, Any]]:
    """Extract text from CSV files."""
    try:
        print(f"Reading CSV file: {file_path}")
        
        # Try different encodings
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise ValueError(f"Could not read CSV file with any of the attempted encodings: {encodings}")
        
        # Convert DataFrame to list of dictionaries
        data = df.replace({np.nan: None}).to_dict('records')
        print(f"Extracted {len(data)} records from CSV file")
        return data
            
    except Exception as e:
        print(f"Error processing CSV file {file_path}: {str(e)}")
        traceback.print_exc()
        return []

def process_privacy_data(raw_data: pd.DataFrame) -> List[Dict[str, Any]]:
    """Process raw privacy/PII data into training examples."""
    training_examples = []
    
    try:
        # Clean column names
        raw_data.columns = [str(col).strip().upper() for col in raw_data.columns]
        required_columns = {'COLUMNNAME', 'PRIVACYTYPE'}
        if not all(col in raw_data.columns for col in required_columns):
            missing = required_columns - set(raw_data.columns)
            print(f"Warning: Missing required columns for privacy data: {missing}")
            return []
        
        # Get list of additional columns for metadata
        metadata_columns = [col for col in raw_data.columns if col not in required_columns]
        print(f"Found additional columns that will be preserved in metadata: {metadata_columns}")
        
        for _, row in raw_data.iterrows():
            try:
                # Build text description
                text_parts = []
                text_parts.append(f"Database Column: {row['COLUMNNAME']}")
                
                if pd.notna(row.get('ENTITYNAME')):
                    text_parts.append(f"Entity: {row['ENTITYNAME']}")
                if pd.notna(row.get('ENTITYDESC')):
                    text_parts.append(f"Description: {row['ENTITYDESC']}")
                if pd.notna(row.get('PRIVACYTYPEDESCRIPTION')):
                    text_parts.append(f"Privacy Description: {row['PRIVACYTYPEDESCRIPTION']}")
                
                text = "\n".join(text_parts)
                
                # Determine privacy classification
                pc_category = "PC1"  # Default to PC1
                if pd.notna(row.get('PRIVACYTYPECLASSIFICATION')):
                    classification = str(row['PRIVACYTYPECLASSIFICATION']).upper()
                    if any(term in classification for term in ["PUBLIC", "DP10", "UNRESTRICTED"]):
                        pc_category = "PC0"
                    elif any(term in classification for term in ["CONFIDENTIAL", "DP30", "HIGHLY RESTRICTED", "SENSITIVE"]):
                        pc_category = "PC3"
                
                # Identify PII types using improved matching
                pii_types = set()
                privacy_type = str(row['PRIVACYTYPE']).strip()
                privacy_name = str(row.get('PRIVACYTYPENAME', '')).strip()
                
                # Find PII types using process.extract
                for text in [privacy_type, privacy_name]:
                    if pd.notna(text):
                        matches = process.extract(
                            text,
                            PII_TYPES,
                            scorer=fuzz.WRatio,
                            score_cutoff=75,  # Higher threshold for PII type matching
                            limit=2,  # Get top 2 matches
                            processor=utils.default_process
                        )
                        
                        for match, score, _ in matches:
                            pii_types.add(match)
                            print(f"Matched PII type: '{text}' -> '{match}' (confidence: {score/100:.2f})")
                
                # Also check the description for additional PII types
                if pd.notna(row.get('PRIVACYTYPEDESCRIPTION')):
                    desc = str(row['PRIVACYTYPEDESCRIPTION'])
                    desc_matches = process.extract(
                        desc,
                        PII_TYPES,
                        scorer=fuzz.token_set_ratio,  # Better for longer text
                        score_cutoff=65,  # Lower threshold for description matching
                        limit=3,  # Get top 3 matches
                        processor=utils.default_process
                    )
                    
                    for match, score, _ in desc_matches:
                        pii_types.add(match)
                        print(f"Matched PII type from description: '{match}' (confidence: {score/100:.2f})")
                
                # If no PII types found through matching, analyze the text
                if not pii_types:
                    _, detected_types, _ = analyze_content_for_pii(text)
                    pii_types.update(detected_types)
                
                # Create example if we have valid PII types
                if pii_types:
                    example = {
                        "type": "pii",
                        "text": text,
                        "pc_category": pc_category,
                        "pii_types": list(pii_types),
                        "metadata": {
                            "source": "raw_data",
                            "original_type": privacy_type,
                            "original_name": privacy_name if pd.notna(row.get('PRIVACYTYPENAME')) else None,
                            "original_classification": row.get('PRIVACYTYPECLASSIFICATION') if pd.notna(row.get('PRIVACYTYPECLASSIFICATION')) else None
                        }
                    }
                    
                    # Add additional metadata
                    for col in metadata_columns:
                        if pd.notna(row.get(col)):
                            example["metadata"][col.lower()] = row[col]
                    
                    training_examples.append(example)
                else:
                    print(f"Warning: No PII types identified for column '{row['COLUMNNAME']}'")
            
            except Exception as e:
                print(f"Error processing privacy row: {str(e)}")
                continue
        
        # Print summary
        print(f"\nProcessed {len(raw_data)} privacy entries:")
        print(f"- Successfully mapped: {len(training_examples)}")
        print(f"- Skipped: {len(raw_data) - len(training_examples)}")
        
        if training_examples:
            # Print mapping statistics
            pii_stats = {}
            pc_stats = {}
            for ex in training_examples:
                pc_stats[ex['pc_category']] = pc_stats.get(ex['pc_category'], 0) + 1
                for pii_type in ex['pii_types']:
                    pii_stats[pii_type] = pii_stats.get(pii_type, 0) + 1
            
            print("\nPrivacy Classification Distribution:")
            for pc, count in sorted(pc_stats.items()):
                print(f"  {pc}: {count} entries")
            
            print("\nTop PII Types:")
            for pii_type, count in sorted(pii_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {pii_type}: {count} occurrences")
    
    except Exception as e:
        print(f"Error processing privacy data: {str(e)}")
        traceback.print_exc()
    
    return training_examples

def main():
    """Main function to demonstrate the enhanced functionality."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Risk & PII Fine-Tuner")
    parser.add_argument("--folder", required=True, help="Folder path containing raw data files")
    parser.add_argument("--output", default="fine_tuning_output", help="Output directory for fine-tuned model")
    parser.add_argument("--training-data", default="training_data", help="Directory to save extracted training data")
    
    args = parser.parse_args()
    
    try:
        # Process folder to extract training data
        print("=== ENHANCED RAW DATA PROCESSING ===")
        training_file = process_folder_for_training_data(args.folder, args.training_data)
        
        print(f"\n=== TRAINING DATA READY ===")
        print(f"Training data saved to: {training_file}")
        
        # Import fine-tuning function from risk_fine_tuner
        from risk_fine_tuner import fine_tune_model
        
        # Start fine-tuning process
        print("\n=== STARTING FINE-TUNING ===")
        result = fine_tune_model(training_file, args.output)
        
        if result:
            print(f"\nFine-tuning completed successfully!")
            print(f"Model saved to: {result}")
            return 0
        else:
            print(f"\nFine-tuning failed!")
            return 1
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    main() 