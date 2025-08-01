#!/usr/bin/env python3
"""
Quick Training Data Verification
Simple script to validate training data format before starting training.
"""

import json
import sys
import os

def verify_training_data(file_path):
    """Verify training data format and return statistics."""
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return False
    
    stats = {
        "total_examples": 0,
        "valid_examples": 0,
        "types": {"risk": 0, "pii": 0, "ddl": 0, "unknown": 0},
        "errors": []
    }
    
    required_fields = {
        "risk": ["type", "text", "l2_category", "macro_risks"],
        "pii": ["type", "text", "pc_category", "pii_types"],
        "ddl": ["type", "ddl_statement", "analysis_result"]
    }
    
    print(f"[VERIFY] Checking: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                stats["total_examples"] += 1
                
                try:
                    example = json.loads(line)
                    example_type = example.get("type", "unknown")
                    
                    # Count types
                    if example_type in stats["types"]:
                        stats["types"][example_type] += 1
                    else:
                        stats["types"]["unknown"] += 1
                    
                    # Validate required fields
                    if example_type in required_fields:
                        missing_fields = []
                        for field in required_fields[example_type]:
                            if field not in example or example[field] is None:
                                missing_fields.append(field)
                        
                        if missing_fields:
                            stats["errors"].append(f"Line {line_num}: Missing fields {missing_fields}")
                        else:
                            # Additional validation
                            if example_type in ["risk", "pii"]:
                                text = example.get("text", "").strip()
                                if not text:
                                    stats["errors"].append(f"Line {line_num}: Empty text field")
                                else:
                                    stats["valid_examples"] += 1
                            elif example_type == "ddl":
                                ddl = example.get("ddl_statement", "").strip()
                                if not ddl:
                                    stats["errors"].append(f"Line {line_num}: Empty DDL statement")
                                else:
                                    stats["valid_examples"] += 1
                    else:
                        stats["errors"].append(f"Line {line_num}: Unknown type '{example_type}'")
                
                except json.JSONDecodeError as e:
                    stats["errors"].append(f"Line {line_num}: Invalid JSON - {str(e)}")
    
    except Exception as e:
        print(f"[ERROR] Failed to read file: {e}")
        return False
    
    # Print results
    print(f"\n[RESULTS] Verification complete")
    print(f"  Total examples: {stats['total_examples']}")
    print(f"  Valid examples: {stats['valid_examples']}")
    print(f"  Success rate: {(stats['valid_examples']/stats['total_examples']*100):.1f}%" if stats['total_examples'] > 0 else "  Success rate: 0%")
    
    print(f"\n[TYPES] Example breakdown:")
    for ex_type, count in stats["types"].items():
        if count > 0:
            print(f"  {ex_type}: {count}")
    
    if stats["errors"]:
        print(f"\n[ERRORS] Found {len(stats['errors'])} issues:")
        for error in stats["errors"][:10]:  # Show first 10 errors
            print(f"  {error}")
        if len(stats["errors"]) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more errors")
    
    # Final verdict
    if stats["valid_examples"] == 0:
        print(f"\n[VERDICT] ❌ FAIL - No valid examples found!")
        print(f"[ACTION] Fix the data format issues above before training")
        return False
    elif stats["valid_examples"] < stats["total_examples"] * 0.8:
        print(f"\n[VERDICT] ⚠️  WARNING - Low success rate ({stats['valid_examples']}/{stats['total_examples']})")
        print(f"[ACTION] Consider fixing issues above for better training")
        return True
    else:
        print(f"\n[VERDICT] ✅ PASS - Data looks good for training!")
        return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python verify_training_data.py <training_file.jsonl>")
        print("\nExample:")
        print("  python verify_training_data.py sample_training_data.jsonl")
        sys.exit(1)
    
    file_path = sys.argv[1]
    success = verify_training_data(file_path)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 