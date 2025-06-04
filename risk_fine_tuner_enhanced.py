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

# Import constants from main file
from risk_fine_tuner import (
    L2,
    MACRO_RISKS,
    PII_PROTECTION_CATEGORIES,
    PII_TYPES,
    PRIVACY_CLASSIFICATIONS,
    SENSITIVITY_LEVELS
)

# Keywords for risk category identification
RISK_KEYWORDS = {
    "1": ["policy", "standard", "governance", "framework", "control", "baseline", "training", "awareness", "maturity", "monitoring", "testing", "kci", "kri", "exception", "tolerance", "issue management"],
    "2": ["development", "acquire", "software", "change", "requirement", "implementation", "dependency", "m&a", "sdlc", "deployment"],
    "3": ["inventory", "asset", "classification", "end of life", "destruction", "hardware", "media", "disposal"],
    "4": ["data identification", "lineage", "data classification", "data governance", "data quality", "metadata"],
    "5": ["encryption", "data loss", "dlp", "logging", "third party", "removable media", "data protection", "at rest", "in transit"],
    "6": ["authentication", "authorization", "privilege", "access", "identity", "joiner", "mover", "leaver", "segregation", "duties", "secrets", "production"],
    "7": ["configuration", "network", "segmentation", "cloud", "data center", "infrastructure"],
    "8": ["vulnerability", "patching", "scanning", "assessment", "s-sdlc", "security testing"],
    "9": ["capacity", "planning", "slo", "availability", "performance", "latency", "monitoring"],
    "10": ["incident", "identification", "classification", "escalation", "trend", "technical"],
    "11": ["security incident", "incident response", "monitoring", "handling", "audit", "post mortem", "threat intelligence"],
    "12": ["resilience", "continuity", "disaster", "recovery", "cyber resilience", "operational"]
}

# Keywords for PII identification
PII_KEYWORDS = {
    "PC0": ["public", "marketing", "documentation", "open data", "website", "brochure"],
    "PC1": ["name", "contact", "business email", "job title", "company", "department", "customer id"],
    "PC3": ["ssn", "social security", "financial", "bank", "credit card", "health", "medical", "password", "credential", "biometric", "national id", "driver license", "passport"]
}

def scan_folder_for_files(folder_path: str) -> List[str]:
    """Scan folder for Excel and CSV files."""
    supported_extensions = ['.xlsx', '.xls', '.csv']
    files_to_process = []
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in supported_extensions):
                files_to_process.append(os.path.join(root, file))
    
    return files_to_process

def analyze_content_for_risk_category(text: str) -> Tuple[str, List[str], float]:
    """
    Analyze text content to identify L2 category and macro risks.
    
    Args:
        text: Text content to analyze
        
    Returns:
        Tuple of (l2_category, macro_risks, confidence_score)
    """
    text_lower = text.lower()
    
    # Initialize variables
    best_l2 = None
    best_l2_score = 0
    identified_risks = set()
    
    # Check each L2 category
    for key, value in L2.items():
        value_lower = value.lower()
        # Calculate similarity score (simple word overlap for now)
        words = set(value_lower.split()) & set(text_lower.split())
        score = len(words) / len(value_lower.split())
        
        if score > best_l2_score:
            best_l2_score = score
            best_l2 = f"{key}. {value}"
    
    # Check for macro risks within the identified L2 category
    if best_l2:
        l2_key = best_l2.split('.')[0]
        for risk in MACRO_RISKS.get(l2_key, []):
            risk_lower = risk.lower()
            if any(word in text_lower for word in risk_lower.split()):
                identified_risks.add(risk)
    
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
    
    for entry in raw_data:
        # Skip empty entries
        if not entry:
            continue
        
        # Try to determine if this is a finding or PII data
        text = ""
        if 'Finding_Title' in entry and 'Finding_Description' in entry:
            # This is a finding
            text = f"Finding: {entry['Finding_Title']}\n\n"
            if entry.get('Finding_Description'):
                text += f"Description: {entry['Finding_Description']}"
            
            # Use provided L2 and macro_risks if available
            if entry.get('L2') and entry.get('macro_risks'):
                example = {
                    "type": "risk",
                    "text": text,
                    "l2_category": entry['L2'],
                    "macro_risks": entry['macro_risks'] if isinstance(entry['macro_risks'], list) else [entry['macro_risks']],
                    "metadata": {
                        "source": "raw_data",
                        "original_l2": entry['L2'],
                        "original_risks": entry['macro_risks']
                    }
                }
            else:
                # Analyze content to determine categories
                l2_category, macro_risks, confidence = analyze_content_for_risk_category(text)
                if confidence > 0.2:  # Only include if we have reasonable confidence
                    example = {
                        "type": "risk",
                        "text": text,
                        "l2_category": l2_category,
                        "macro_risks": macro_risks,
                        "metadata": {
                            "source": "analyzed",
                            "confidence": confidence
                        }
                    }
                else:
                    continue
                    
            training_examples.append(example)
            
        elif 'COLUMNNAME' in entry and 'PRIVACYTYPE' in entry:
            # This is PII data
            text = f"Database Column: {entry['COLUMNNAME']}\n"
            if entry.get('ENTITYNAME'):
                text += f"Entity: {entry['ENTITYNAME']}\n"
            if entry.get('ENTITYDESC'):
                text += f"Description: {entry['ENTITYDESC']}\n"
            if entry.get('PRIVACYTYPEDESCRIPTION'):
                text += f"Privacy Description: {entry['PRIVACYTYPEDESCRIPTION']}\n"
            
            # Use provided classification if available
            if entry.get('PRIVACYTYPECLASSIFICATION'):
                classification = str(entry['PRIVACYTYPECLASSIFICATION']).upper()
                if "PUBLIC" in classification or "DP10" in classification:
                    pc_category = "PC0"
                elif any(level in classification for level in ["CONFIDENTIAL", "DP30", "HIGHLY RESTRICTED"]):
                    pc_category = "PC3"
                else:
                    pc_category = "PC1"
                
                example = {
                    "type": "pii",
                    "text": text,
                    "pc_category": pc_category,
                    "pii_types": [],  # Will be filled by analysis
                    "metadata": {
                        "source": "raw_data",
                        "original_classification": entry['PRIVACYTYPECLASSIFICATION'],
                        "privacy_type": entry.get('PRIVACYTYPE'),
                        "privacy_name": entry.get('PRIVACYTYPENAME')
                    }
                }
            else:
                # Analyze content to determine PII
                pc_category, pii_types, confidence = analyze_content_for_pii(text)
                if confidence > 0.2:  # Only include if we have reasonable confidence
                    example = {
                        "type": "pii",
                        "text": text,
                        "pc_category": pc_category,
                        "pii_types": pii_types,
                        "metadata": {
                            "source": "analyzed",
                            "confidence": confidence
                        }
                    }
                else:
                    continue
                    
            training_examples.append(example)
    
    return training_examples

def process_folder_for_training_data(folder_path: str, output_dir: str = "training_data") -> str:
    """
    Process all files in a folder to extract training data.
    
    Args:
        folder_path: Path to the folder containing raw data files
        output_dir: Directory to save processed training data
        
    Returns:
        Path to the generated training data file
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Scan folder for files
    files_to_process = scan_folder_for_files(folder_path)
    
    if not files_to_process:
        raise ValueError(f"No supported files found in folder: {folder_path}")
    
    # Extract raw data from all files
    all_raw_data = []
    
    for file_path in files_to_process:
        print(f"\nProcessing file: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in ['.xlsx', '.xls']:
            raw_data = extract_text_from_excel(file_path)
        elif file_ext == '.csv':
            raw_data = extract_text_from_csv(file_path)
        else:
            print(f"Skipping unsupported file type: {file_ext}")
            continue
        
        all_raw_data.extend(raw_data)
        print(f"Extracted {len(raw_data)} entries from {os.path.basename(file_path)}")
    
    print(f"\nTotal raw data entries extracted: {len(all_raw_data)}")
    
    # Convert raw data to training examples
    training_examples = process_raw_data_to_training_examples(all_raw_data)
    
    if not training_examples:
        raise ValueError("No training examples could be generated from the raw data")
    
    # Save training examples to JSONL file
    training_file_path = os.path.join(output_dir, "auto_generated_training_data.jsonl")
    
    with open(training_file_path, 'w', encoding='utf-8') as f:
        for example in training_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"\nSaved {len(training_examples)} training examples to: {training_file_path}")
    
    # Also save a summary report
    summary_path = os.path.join(output_dir, "extraction_summary.json")
    summary = {
        "total_files_processed": len(files_to_process),
        "total_raw_entries": len(all_raw_data),
        "total_training_examples": len(training_examples),
        "risk_examples": len([ex for ex in training_examples if ex["type"] == "risk"]),
        "pii_examples": len([ex for ex in training_examples if ex["type"] == "pii"]),
        "processed_files": [os.path.basename(f) for f in files_to_process],
        "extraction_timestamp": pd.Timestamp.now().isoformat()
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved extraction summary to: {summary_path}")
    
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

def process_findings_data(raw_data: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Process raw findings data into training examples.
    
    Key columns:
    - Finding_Title: Title of the finding
    - Finding_Description: Detailed description of the finding
    - L2: L2 category (maps to our L2 categories)
    - macro_risks: List of macro risks (maps to our MACRO_RISKS)
    
    Additional columns will be preserved in metadata.
    """
    training_examples = []
    
    try:
        # Clean column names
        raw_data.columns = [str(col).strip().upper() for col in raw_data.columns]
        required_columns = {'FINDING_TITLE', 'FINDING_DESCRIPTION', 'L2', 'MACRO_RISKS'}
        if not all(col in raw_data.columns for col in required_columns):
            missing = required_columns - set(raw_data.columns)
            print(f"Warning: Missing required columns: {missing}")
            return []
        
        # Get list of additional columns for metadata
        metadata_columns = [col for col in raw_data.columns if col not in required_columns]
        print(f"Found additional columns that will be preserved in metadata: {metadata_columns}")
        
        for _, row in raw_data.iterrows():
            try:
                # Create context from title and description
                context = f"Finding: {row['FINDING_TITLE']}\n\n"
                if pd.notna(row['FINDING_DESCRIPTION']):
                    context += f"Description: {row['FINDING_DESCRIPTION']}"
                
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
                            if l2_value.lower() in value.lower() or value.lower() in l2_value.lower():
                                l2_category = f"{key}. {value}"
                                print(f"Fuzzy matched L2 category: '{l2_value}' -> '{value}'")
                                break
                
                # Parse macro risks
                macro_risks = set()
                if pd.notna(row['MACRO_RISKS']):
                    # Handle different formats (string, list, comma-separated, semicolon-separated)
                    risks = row['MACRO_RISKS']
                    if isinstance(risks, str):
                        # Try different delimiters
                        if ',' in risks:
                            risks = [r.strip() for r in risks.split(',')]
                        elif ';' in risks:
                            risks = [r.strip() for r in risks.split(';')]
                        elif '\n' in risks:
                            risks = [r.strip() for r in risks.split('\n')]
                        else:
                            risks = [risks.strip()]
                    elif isinstance(risks, list):
                        risks = [str(r).strip() for r in risks]
                    
                    # Match with defined macro risks
                    for risk in risks:
                        risk = risk.strip()
                        matched = False
                        
                        # First try exact match
                        for themes in MACRO_RISKS.values():
                            if risk in themes:
                                macro_risks.add(risk)
                                matched = True
                                break
                        
                        # If no exact match, try fuzzy match
                        if not matched:
                            best_match = None
                            best_score = 0
                            risk_lower = risk.lower()
                            
                            for themes in MACRO_RISKS.values():
                                for theme in themes:
                                    theme_lower = theme.lower()
                                    # Calculate similarity score
                                    words_risk = set(risk_lower.split())
                                    words_theme = set(theme_lower.split())
                                    common_words = words_risk & words_theme
                                    if common_words:
                                        score = len(common_words) / max(len(words_risk), len(words_theme))
                                        if score > best_score and score > 0.5:  # Threshold for fuzzy matching
                                            best_score = score
                                            best_match = theme
                            
                            if best_match:
                                macro_risks.add(best_match)
                                print(f"Fuzzy matched macro risk: '{risk}' -> '{best_match}'")
                
                # Collect metadata from additional columns
                metadata = {
                    "source": "raw_data",
                    "original_l2": row['L2'] if pd.notna(row['L2']) else None,
                    "original_risks": row['MACRO_RISKS'] if pd.notna(row['MACRO_RISKS']) else None,
                }
                
                # Add additional columns to metadata
                for col in metadata_columns:
                    if pd.notna(row[col]):
                        metadata[col.lower()] = row[col]
                
                # Only create example if we have valid L2 and macro risks
                if l2_category and macro_risks:
                    example = {
                        "type": "risk",
                        "text": context,
                        "l2_category": l2_category,
                        "macro_risks": list(macro_risks),  # Convert set back to list
                        "metadata": metadata
                    }
                    training_examples.append(example)
                else:
                    print(f"Warning: Skipping finding '{row['FINDING_TITLE']}' due to:")
                    if not l2_category:
                        print(f"  - Could not map L2 category: '{row['L2'] if pd.notna(row['L2']) else 'N/A'}'")
                    if not macro_risks:
                        print(f"  - Could not map macro risks: '{row['MACRO_RISKS'] if pd.notna(row['MACRO_RISKS']) else 'N/A'}'")
            
            except Exception as e:
                print(f"Error processing row: {str(e)}")
                continue
        
        # Print summary
        print(f"\nProcessed {len(raw_data)} findings:")
        print(f"- Successfully mapped: {len(training_examples)}")
        print(f"- Skipped: {len(raw_data) - len(training_examples)}")
        
        if training_examples:
            # Print mapping statistics
            l2_stats = {}
            risk_stats = {}
            for ex in training_examples:
                l2 = ex['l2_category'].split('.')[0]
                l2_stats[l2] = l2_stats.get(l2, 0) + 1
                for risk in ex['macro_risks']:
                    risk_stats[risk] = risk_stats.get(risk, 0) + 1
            
            print("\nL2 Category Distribution:")
            for l2, count in sorted(l2_stats.items()):
                print(f"  L2 {l2}: {count} findings")
            
            print("\nTop Macro Risks:")
            for risk, count in sorted(risk_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {risk}: {count} occurrences")
        
    except Exception as e:
        print(f"Error processing findings data: {str(e)}")
        traceback.print_exc()
    
    return training_examples

def extract_text_from_excel(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from Excel files, with special handling for findings data.
    Supports both findings data (with Finding_Title, Finding_Description, etc.)
    and privacy data (with COLUMNNAME, PRIVACYTYPE, etc.).
    """
    try:
        print(f"Reading Excel file: {file_path}")
        
        # Try to read all sheets
        all_sheets = pd.read_excel(file_path, sheet_name=None)
        training_examples = []
        
        for sheet_name, df in all_sheets.items():
            print(f"\nProcessing sheet: {sheet_name}")
            
            # Clean column names
            df.columns = [str(col).strip().upper() for col in df.columns]
            
            # Check if this is findings data
            if 'FINDING_TITLE' in df.columns:
                print(f"Found findings data in sheet {sheet_name}")
                training_examples.extend(process_findings_data(df))
                continue
            
            # Check if this is privacy data
            if 'COLUMNNAME' in df.columns:
                print(f"Found privacy data in sheet {sheet_name}")
                training_examples.extend(process_privacy_data(df))
                continue
            
            print(f"Warning: Unknown data format in sheet {sheet_name}")
            print(f"Available columns: {', '.join(df.columns)}")
            print("Expected either:")
            print("- Findings data columns: FINDING_TITLE, FINDING_DESCRIPTION, L2, MACRO_RISKS")
            print("- Privacy data columns: COLUMNNAME, PRIVACYTYPE, PRIVACYTYPENAME")
        
        if not training_examples:
            print(f"Warning: No valid training examples found in {file_path}")
        else:
            print(f"\nExtracted {len(training_examples)} training examples from {file_path}")
            risk_count = sum(1 for ex in training_examples if ex["type"] == "risk")
            pii_count = sum(1 for ex in training_examples if ex["type"] == "pii")
            print(f"- Risk examples: {risk_count}")
            print(f"- PII examples: {pii_count}")
        
        return training_examples
            
    except Exception as e:
        print(f"Error processing Excel file {file_path}: {str(e)}")
        traceback.print_exc()
        return []

def extract_text_from_csv(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from CSV files, with special handling for findings data.
    Supports both findings data (with Finding_Title, Finding_Description, etc.)
    and privacy data (with COLUMNNAME, PRIVACYTYPE, etc.).
    """
    try:
        print(f"Reading CSV file: {file_path}")
        
        # Try to read with different encodings
        encodings = ['utf-8', 'latin1', 'iso-8859-1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"Successfully read file with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise ValueError(f"Could not read CSV file with any of the attempted encodings: {encodings}")
        
        # Clean column names
        df.columns = [str(col).strip().upper() for col in df.columns]
        
        # Check if this is findings data
        if 'FINDING_TITLE' in df.columns:
            print("Found findings data")
            return process_findings_data(df)
        
        # Check if this is privacy data
        if 'COLUMNNAME' in df.columns:
            print("Found privacy data")
            return process_privacy_data(df)
        
        print("Warning: Unknown data format")
        print(f"Available columns: {', '.join(df.columns)}")
        print("Expected either:")
        print("- Findings data columns: FINDING_TITLE, FINDING_DESCRIPTION, L2, MACRO_RISKS")
        print("- Privacy data columns: COLUMNNAME, PRIVACYTYPE, PRIVACYTYPENAME")
        return []
            
    except Exception as e:
        print(f"Error processing CSV file {file_path}: {str(e)}")
        traceback.print_exc()
        return []

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
        print(f"You can now proceed with fine-tuning using this data.")
        
        # Note: The actual fine-tuning code would continue here...
        # For now, we'll just show the data extraction capability
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 