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
    """
    Scan a folder and return all Excel, CSV, and JSON files for processing.
    
    Args:
        folder_path: Path to the folder to scan
        
    Returns:
        List of file paths to process
    """
    supported_extensions = ['.xlsx', '.xls', '.csv', '.json', '.jsonl']
    files_to_process = []
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist")
        return []
    
    print(f"Scanning folder: {folder_path}")
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            if file_ext in supported_extensions:
                files_to_process.append(file_path)
                print(f"Found file: {file}")
    
    print(f"Found {len(files_to_process)} files to process")
    return files_to_process

def extract_text_from_excel(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract text content from Excel files and create raw data entries.
    
    Args:
        file_path: Path to the Excel file
        
    Returns:
        List of extracted text entries with metadata
    """
    extracted_data = []
    
    try:
        # Read all sheets from the Excel file
        excel_file = pd.ExcelFile(file_path)
        print(f"Processing Excel file: {file_path}")
        print(f"Found sheets: {excel_file.sheet_names}")
        
        for sheet_name in excel_file.sheet_names:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                print(f"Processing sheet '{sheet_name}' with {len(df)} rows and {len(df.columns)} columns")
                
                # Extract column headers as context
                headers = [str(col) for col in df.columns if not str(col).startswith('Unnamed')]
                
                # Process each row
                for index, row in df.iterrows():
                    # Combine all non-null values in the row as text
                    row_text_parts = []
                    row_data = {}
                    
                    for col_index, value in enumerate(row):
                        if pd.notna(value) and str(value).strip():
                            col_name = headers[col_index] if col_index < len(headers) else f"Column_{col_index}"
                            text_value = str(value).strip()
                            row_text_parts.append(f"{col_name}: {text_value}")
                            row_data[col_name] = text_value
                    
                    if row_text_parts:
                        combined_text = " | ".join(row_text_parts)
                        
                        extracted_data.append({
                            "text": combined_text,
                            "source_file": os.path.basename(file_path),
                            "sheet_name": sheet_name,
                            "row_number": index + 2,  # +2 for 1-indexing and header
                            "headers": headers,
                            "raw_data": row_data
                        })
            
            except Exception as e:
                print(f"Error processing sheet '{sheet_name}': {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error processing Excel file {file_path}: {str(e)}")
        
    return extracted_data

def extract_text_from_csv(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract text content from CSV files and create raw data entries.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        List of extracted text entries with metadata
    """
    extracted_data = []
    
    try:
        print(f"Processing CSV file: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            # Try to detect if there are headers
            sample = f.read(1024)
            f.seek(0)
            sniffer = csv.Sniffer()
            has_header = sniffer.has_header(sample)
            
            reader = csv.reader(f)
            
            # Read headers if they exist
            headers = []
            if has_header:
                headers = next(reader)
            else:
                # Create default headers
                first_row = next(reader)
                headers = [f"Column_{i}" for i in range(len(first_row))]
                # Process the first row since we already read it
                if first_row:
                    row_text_parts = []
                    row_data = {}
                    
                    for col_index, value in enumerate(first_row):
                        if value and value.strip():
                            col_name = headers[col_index] if col_index < len(headers) else f"Column_{col_index}"
                            text_value = value.strip()
                            row_text_parts.append(f"{col_name}: {text_value}")
                            row_data[col_name] = text_value
                    
                    if row_text_parts:
                        combined_text = " | ".join(row_text_parts)
                        extracted_data.append({
                            "text": combined_text,
                            "source_file": os.path.basename(file_path),
                            "row_number": 1,
                            "headers": headers,
                            "raw_data": row_data
                        })
            
            # Process remaining rows
            for row_index, row in enumerate(reader):
                if not row:
                    continue
                    
                row_text_parts = []
                row_data = {}
                
                for col_index, value in enumerate(row):
                    if value and value.strip():
                        col_name = headers[col_index] if col_index < len(headers) else f"Column_{col_index}"
                        text_value = value.strip()
                        row_text_parts.append(f"{col_name}: {text_value}")
                        row_data[col_name] = text_value
                
                if row_text_parts:
                    combined_text = " | ".join(row_text_parts)
                    extracted_data.append({
                        "text": combined_text,
                        "source_file": os.path.basename(file_path),
                        "row_number": row_index + (2 if has_header else 1),
                        "headers": headers,
                        "raw_data": row_data
                    })
                    
    except Exception as e:
        print(f"Error processing CSV file {file_path}: {str(e)}")
        
    return extracted_data

def analyze_content_for_risk_category(text: str) -> Tuple[Optional[str], List[str], float]:
    """
    Analyze text content to identify the most likely risk category and themes.
    
    Args:
        text: Text content to analyze
        
    Returns:
        Tuple of (macro_risk_id, risk_themes, confidence_score)
    """
    text_lower = text.lower()
    category_scores = {}
    
    # Score each category based on keyword matches
    for category_id, keywords in RISK_KEYWORDS.items():
        score = 0
        matched_keywords = []
        
        for keyword in keywords:
            if keyword in text_lower:
                score += 1
                matched_keywords.append(keyword)
        
        if score > 0:
            # Normalize score by number of keywords in category
            normalized_score = score / len(keywords)
            category_scores[category_id] = {
                "score": normalized_score,
                "matched_keywords": matched_keywords,
                "raw_score": score
            }
    
    if not category_scores:
        return None, [], 0.0
    
    # Find the best matching category
    best_category = max(category_scores.keys(), key=lambda x: category_scores[x]["score"])
    best_score = category_scores[best_category]["score"]
    
    # Map matched keywords to thematic risks
    matched_themes = []
    available_themes = THEMATIC_RISKS[best_category]
    
    for theme in available_themes:
        theme_lower = theme.lower()
        # Check if any part of the theme appears in the text
        theme_words = theme_lower.split()
        if any(word in text_lower for word in theme_words if len(word) > 3):
            matched_themes.append(theme)
    
    # If no themes matched, try to infer from the best matching keywords
    if not matched_themes:
        matched_keywords = category_scores[best_category]["matched_keywords"]
        
        # Map keywords to potential themes (simplified heuristic)
        if best_category == "1" and any(kw in matched_keywords for kw in ["policy", "standard", "governance"]):
            matched_themes.append("Policy/Standard Review")
        elif best_category == "2" and any(kw in matched_keywords for kw in ["change", "implementation"]):
            matched_themes.append("Change Process (Standards & Emergency)")
        elif best_category == "3" and any(kw in matched_keywords for kw in ["inventory", "asset"]):
            matched_themes.append("Inventory Accuracy & Completeness")
        elif best_category == "4" and any(kw in matched_keywords for kw in ["data"]):
            matched_themes.append("Data Identification, Inventory & Lineage")
        elif best_category == "5" and any(kw in matched_keywords for kw in ["encryption", "protection"]):
            matched_themes.append("Encryption (At Rest, Use, Transit)")
        elif best_category == "6" and any(kw in matched_keywords for kw in ["access", "authentication"]):
            matched_themes.append("Authentication")
        elif best_category == "7" and any(kw in matched_keywords for kw in ["configuration", "infrastructure"]):
            matched_themes.append("Configuration Management")
        elif best_category == "8" and any(kw in matched_keywords for kw in ["vulnerability", "patching"]):
            matched_themes.append("Vulnerability assessment and risk treatment")
        elif best_category == "9" and any(kw in matched_keywords for kw in ["capacity", "monitoring"]):
            matched_themes.append("Capacity Planning")
        elif best_category == "10" and any(kw in matched_keywords for kw in ["incident"]):
            matched_themes.append("Incident Identification & Classification")
        elif best_category == "11" and any(kw in matched_keywords for kw in ["security incident", "response"]):
            matched_themes.append("Incident Response Planning")
        elif best_category == "12" and any(kw in matched_keywords for kw in ["resilience", "continuity"]):
            matched_themes.append("Operational Resiliency")
    
    # Ensure we have at least one theme
    if not matched_themes and best_category in THEMATIC_RISKS:
        # Use the first theme as a fallback
        matched_themes.append(THEMATIC_RISKS[best_category][0])
    
    return best_category, matched_themes, best_score

def analyze_content_for_pii(text: str) -> Tuple[str, List[str], float]:
    """
    Analyze text content to identify PII and classify protection category.
    
    Args:
        text: Text content to analyze
        
    Returns:
        Tuple of (pc_category, pii_types, confidence_score)
    """
    text_lower = text.lower()
    
    # Check for high-sensitivity PII (PC3)
    pc3_indicators = 0
    pc3_types = []
    
    for pii_type in PII_TYPES:
        type_lower = pii_type.lower()
        if type_lower in text_lower:
            if pii_type in ["SSN", "Financial", "Health", "Credentials", "Biometric", "National ID"]:
                pc3_indicators += 1
                pc3_types.append(pii_type)
    
    # Specific pattern checks for PC3
    patterns_pc3 = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
        r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card pattern
        r'\bpassword\b', r'\bcredential\b', r'\bauth\b',
        r'\bmedical\b', r'\bhealth\b', r'\bhipaa\b',
        r'\bbiometric\b', r'\bfingerprint\b'
    ]
    
    for pattern in patterns_pc3:
        if re.search(pattern, text_lower):
            pc3_indicators += 1
            if pattern.startswith(r'\bpassword') or pattern.startswith(r'\bcredential'):
                pc3_types.append("Credentials")
            elif pattern.startswith(r'\bmedical') or pattern.startswith(r'\bhealth'):
                pc3_types.append("Health")
            elif pattern.startswith(r'\bbiometric'):
                pc3_types.append("Biometric")
            elif pattern.startswith(r'\b\d{3}-\d{2}'):
                pc3_types.append("SSN")
            elif pattern.startswith(r'\b\d{4}'):
                pc3_types.append("Financial")
    
    if pc3_indicators > 0:
        return "PC3", list(set(pc3_types)), min(1.0, pc3_indicators * 0.3)
    
    # Check for medium-sensitivity PII (PC1)
    pc1_indicators = 0
    pc1_types = []
    
    for pii_type in PII_TYPES:
        type_lower = pii_type.lower()
        if type_lower in text_lower:
            if pii_type in ["Name", "Email", "Phone", "Address", "Customer ID", "Employment"]:
                pc1_indicators += 1
                pc1_types.append(pii_type)
    
    # Specific pattern checks for PC1
    patterns_pc1 = [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email pattern
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone pattern
        r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'     # Name pattern
    ]
    
    for pattern in patterns_pc1:
        if re.search(pattern, text_lower):
            pc1_indicators += 1
            if pattern.startswith(r'\b[A-Za-z0-9._%+-]+@'):
                pc1_types.append("Email")
            elif pattern.startswith(r'\b\d{3}'):
                pc1_types.append("Phone")
            elif pattern.startswith(r'\b[A-Z][a-z]+ [A-Z]'):
                pc1_types.append("Name")
    
    if pc1_indicators > 0:
        return "PC1", list(set(pc1_types)), min(1.0, pc1_indicators * 0.2)
    
    # Default to PC0 (Public)
    return "PC0", [], 0.1

def process_raw_data_to_training_examples(raw_data_entries: List[Dict[str, Any]], min_confidence: float = 0.1) -> List[Dict[str, Any]]:
    """
    Process raw data entries and convert them to training examples.
    
    Args:
        raw_data_entries: List of raw data entries from files
        min_confidence: Minimum confidence threshold for including examples
        
    Returns:
        List of training examples
    """
    training_examples = []
    
    print(f"Processing {len(raw_data_entries)} raw data entries...")
    
    for entry in tqdm(raw_data_entries, desc="Analyzing content"):
        text = entry["text"]
        
        # Skip very short text entries
        if len(text.strip()) < 20:
            continue
        
        # Analyze for risk categorization
        risk_category, risk_themes, risk_confidence = analyze_content_for_risk_category(text)
        
        if risk_category and risk_themes and risk_confidence >= min_confidence:
            macro_risk = f"{risk_category}. {MACRO_RISKS[risk_category]}"
            
            training_examples.append({
                "type": "risk",
                "text": text,
                "macro_risk": macro_risk,
                "risk_themes": risk_themes,
                "confidence": risk_confidence,
                "source_file": entry["source_file"],
                "metadata": {
                    "sheet_name": entry.get("sheet_name"),
                    "row_number": entry.get("row_number"),
                    "extraction_method": "automated_analysis"
                }
            })
        
        # Analyze for PII classification
        pc_category, pii_types, pii_confidence = analyze_content_for_pii(text)
        
        if pii_confidence >= min_confidence:
            training_examples.append({
                "type": "pii",
                "text": text,
                "pc_category": pc_category,
                "pii_types": pii_types,
                "confidence": pii_confidence,
                "source_file": entry["source_file"],
                "metadata": {
                    "sheet_name": entry.get("sheet_name"),
                    "row_number": entry.get("row_number"),
                    "extraction_method": "automated_analysis"
                }
            })
    
    print(f"Generated {len(training_examples)} training examples from raw data")
    
    # Print summary statistics
    risk_examples = [ex for ex in training_examples if ex["type"] == "risk"]
    pii_examples = [ex for ex in training_examples if ex["type"] == "pii"]
    
    print(f"Risk examples: {len(risk_examples)}")
    print(f"PII examples: {len(pii_examples)}")
    
    if risk_examples:
        risk_categories = {}
        for ex in risk_examples:
            cat = ex["macro_risk"].split(".")[0]
            risk_categories[cat] = risk_categories.get(cat, 0) + 1
        print("Risk category distribution:")
        for cat, count in sorted(risk_categories.items()):
            print(f"  Category {cat}: {count} examples")
    
    if pii_examples:
        pii_categories = {}
        for ex in pii_examples:
            cat = ex["pc_category"]
            pii_categories[cat] = pii_categories.get(cat, 0) + 1
        print("PII category distribution:")
        for cat, count in sorted(pii_categories.items()):
            print(f"  {cat}: {count} examples")
    
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