#!/usr/bin/env python3
"""
Debug Training Data - Diagnostic tool for training data issues
Helps identify why training examples are being skipped or malformed.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import traceback

# Import necessary modules
try:
    from risk_fine_tuner_enhanced import load_training_file, detect_data_format
    from risk_fine_tuner_gradient_fixed import (
        format_all_categories_for_prompt,
        format_risk_example,
        format_pii_example, 
        format_ddl_example,
        process_batch
    )
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    print("Make sure you're running this from the correct directory")
    sys.exit(1)

def analyze_training_examples(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze training examples to identify issues."""
    analysis = {
        "total_examples": len(examples),
        "example_types": {},
        "missing_fields": {},
        "sample_examples": [],
        "errors": []
    }
    
    for i, example in enumerate(examples[:10]):  # Analyze first 10 examples
        try:
            # Check example type
            example_type = example.get("type", "MISSING")
            analysis["example_types"][example_type] = analysis["example_types"].get(example_type, 0) + 1
            
            # Check required fields based on type
            required_fields = {
                "risk": ["text", "l2_category", "macro_risks"],
                "pii": ["text", "pc_category", "pii_types"],
                "ddl": ["ddl_statement", "analysis_result"]
            }
            
            if example_type in required_fields:
                missing = []
                for field in required_fields[example_type]:
                    if field not in example or example[field] is None:
                        missing.append(field)
                
                if missing:
                    key = f"{example_type}_missing_fields"
                    if key not in analysis["missing_fields"]:
                        analysis["missing_fields"][key] = {}
                    for field in missing:
                        analysis["missing_fields"][key][field] = analysis["missing_fields"][key].get(field, 0) + 1
            
            # Store sample for inspection
            if len(analysis["sample_examples"]) < 3:
                analysis["sample_examples"].append({
                    "index": i,
                    "type": example_type,
                    "keys": list(example.keys()),
                    "sample_data": {k: str(v)[:100] + "..." if len(str(v)) > 100 else v 
                                   for k, v in example.items()}
                })
                
        except Exception as e:
            analysis["errors"].append(f"Example {i}: {str(e)}")
    
    return analysis

def test_example_formatting(examples: List[Dict[str, Any]], max_test: int = 5) -> Dict[str, Any]:
    """Test formatting function on sample examples."""
    categories = format_all_categories_for_prompt()
    results = {
        "successful_formats": 0,
        "failed_formats": 0,
        "format_errors": [],
        "sample_outputs": []
    }
    
    test_examples = examples[:max_test]
    
    for i, example in enumerate(test_examples):
        try:
            example_type = example.get("type", "unknown")
            formatted_example = None
            
            if example_type == "risk":
                formatted_example = format_risk_example(example, categories)
            elif example_type == "pii":
                formatted_example = format_pii_example(example, categories)
            elif example_type == "ddl":
                formatted_example = format_ddl_example(example, categories)
            else:
                results["format_errors"].append(f"Example {i}: Unknown type '{example_type}'")
                results["failed_formats"] += 1
                continue
            
            if formatted_example:
                results["successful_formats"] += 1
                if len(results["sample_outputs"]) < 2:
                    results["sample_outputs"].append({
                        "index": i,
                        "type": example_type,
                        "formatted_length": len(json.dumps(formatted_example)),
                        "has_messages": "messages" in formatted_example
                    })
            else:
                results["failed_formats"] += 1
                results["format_errors"].append(f"Example {i}: Formatting returned None")
                
        except Exception as e:
            results["failed_formats"] += 1
            results["format_errors"].append(f"Example {i}: {str(e)}")
            traceback.print_exc()
    
    return results

def create_fixed_training_file(examples: List[Dict[str, Any]], output_file: str) -> Dict[str, Any]:
    """Create a fixed training file with better error handling."""
    stats = {
        "total_input": len(examples),
        "successful_writes": 0,
        "skipped_examples": 0,
        "errors": []
    }
    
    categories = format_all_categories_for_prompt()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, example in enumerate(examples):
            try:
                example_type = example.get("type", "unknown")
                
                # Add type if missing but can be inferred
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
                
                formatted_example = None
                
                if example_type == "risk":
                    formatted_example = format_risk_example(example, categories)
                elif example_type == "pii":
                    formatted_example = format_pii_example(example, categories)
                elif example_type == "ddl":
                    formatted_example = format_ddl_example(example, categories)
                else:
                    stats["errors"].append(f"Example {i}: Unknown type '{example_type}'")
                    stats["skipped_examples"] += 1
                    continue
                
                if formatted_example:
                    f.write(json.dumps(formatted_example) + '\n')
                    stats["successful_writes"] += 1
                else:
                    stats["errors"].append(f"Example {i}: Formatting returned None for type '{example_type}'")
                    stats["skipped_examples"] += 1
                    
            except Exception as e:
                stats["errors"].append(f"Example {i}: {str(e)}")
                stats["skipped_examples"] += 1
    
    return stats

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_training_data.py <training_data_path> [output_file]")
        print("\nExamples:")
        print("  python debug_training_data.py ./my_data")
        print("  python debug_training_data.py ./my_data ./fixed_train.jsonl")
        sys.exit(1)
    
    training_data_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("=== TRAINING DATA DIAGNOSTICS ===")
    print(f"[INPUT] Data path: {training_data_path}")
    
    # Check if path exists
    if not os.path.exists(training_data_path):
        print(f"[ERROR] Path does not exist: {training_data_path}")
        sys.exit(1)
    
    # Detect data format
    from risk_fine_tuner_gradient_fixed import detect_data_format
    data_format = detect_data_format(training_data_path)
    print(f"[FORMAT] Detected format: {data_format}")
    
    # Load training examples
    print("\n=== LOADING TRAINING DATA ===")
    training_examples = []
    
    try:
        if data_format == 'raw_folder':
            from risk_fine_tuner_enhanced import process_folder_for_training_data
            temp_training_file = process_folder_for_training_data(training_data_path, "./debug_temp")
            training_examples = load_training_file(temp_training_file)
        elif data_format in ['raw_file', 'formatted_file']:
            training_examples = load_training_file(training_data_path)
        elif os.path.isdir(training_data_path):
            for filename in os.listdir(training_data_path):
                file_path = os.path.join(training_data_path, filename)
                if os.path.isfile(file_path):
                    training_examples.extend(load_training_file(file_path))
        
        print(f"[SUCCESS] Loaded {len(training_examples)} examples")
        
    except Exception as e:
        print(f"[ERROR] Failed to load training data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if not training_examples:
        print("[ERROR] No training examples found!")
        sys.exit(1)
    
    # Analyze examples
    print("\n=== ANALYZING EXAMPLES ===")
    analysis = analyze_training_examples(training_examples)
    
    print(f"[STATS] Total examples: {analysis['total_examples']}")
    print(f"[TYPES] Example types found:")
    for ex_type, count in analysis["example_types"].items():
        print(f"  - {ex_type}: {count}")
    
    if analysis["missing_fields"]:
        print(f"[ISSUES] Missing fields detected:")
        for issue, fields in analysis["missing_fields"].items():
            print(f"  - {issue}: {fields}")
    
    if analysis["errors"]:
        print(f"[ERRORS] Analysis errors:")
        for error in analysis["errors"][:5]:  # Show first 5 errors
            print(f"  - {error}")
    
    # Test formatting
    print("\n=== TESTING FORMATTING ===")
    format_results = test_example_formatting(training_examples)
    
    print(f"[FORMATTING] Results:")
    print(f"  - Successful: {format_results['successful_formats']}")
    print(f"  - Failed: {format_results['failed_formats']}")
    
    if format_results["format_errors"]:
        print(f"[FORMAT ERRORS] First 5 errors:")
        for error in format_results["format_errors"][:5]:
            print(f"  - {error}")
    
    # Show sample examples
    print("\n=== SAMPLE EXAMPLES ===")
    for sample in analysis["sample_examples"]:
        print(f"[SAMPLE {sample['index']}] Type: {sample['type']}")
        print(f"  Keys: {sample['keys']}")
        print(f"  Data preview: {sample['sample_data']}")
        print()
    
    # Create fixed file if requested
    if output_file:
        print(f"\n=== CREATING FIXED FILE ===")
        print(f"[OUTPUT] Writing to: {output_file}")
        
        stats = create_fixed_training_file(training_examples, output_file)
        
        print(f"[RESULTS] File creation stats:")
        print(f"  - Input examples: {stats['total_input']}")
        print(f"  - Successfully written: {stats['successful_writes']}")
        print(f"  - Skipped: {stats['skipped_examples']}")
        
        if stats["errors"]:
            print(f"[ERRORS] Creation errors (first 5):")
            for error in stats["errors"][:5]:
                print(f"  - {error}")
        
        if stats["successful_writes"] > 0:
            print(f"[SUCCESS] Fixed training file created: {output_file}")
            print(f"[USAGE] You can now use this file for training")
        else:
            print(f"[ERROR] No examples were successfully written!")
    
    print("\n=== DIAGNOSIS COMPLETE ===")

if __name__ == "__main__":
    main() 