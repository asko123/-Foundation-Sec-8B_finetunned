#!/usr/bin/env python3
"""
Test script to verify the integration between risk_fine_tuner.py and risk_inference.py
"""

import os
import json
import sys

def create_test_data():
    """Create minimal test data for quick testing."""
    test_data = [
        # Risk examples
        {
            "type": "risk",
            "text": "The database server has not been patched in 6 months, leaving it vulnerable to known exploits.",
            "macro_risk": "8. Manage IT Vulnerabilities & Patching",
            "risk_themes": ["Patching Completeness", "Vulnerability assessment and risk treatment"]
        },
        {
            "type": "risk", 
            "text": "User accounts are not being properly deactivated when employees leave the company.",
            "macro_risk": "6. Identity & Access Management",
            "risk_themes": ["Identity Access Lifecycle (Joiners/Movers/Leavers)"]
        },
        # PII examples
        {
            "type": "pii",
            "text": "Customer email: john.doe@example.com, Phone: 555-123-4567",
            "pc_category": "PC1",
            "pii_types": ["Email", "Phone"]
        },
        {
            "type": "pii",
            "text": "Patient SSN: 123-45-6789, Medical Record: Diabetes diagnosis",
            "pc_category": "PC3",
            "pii_types": ["SSN", "Health"]
        }
    ]
    
    # Save as JSONL
    with open("test_data.jsonl", "w") as f:
        for example in test_data:
            f.write(json.dumps(example) + "\n")
    
    print("✓ Created test_data.jsonl with 4 examples (2 risk, 2 PII)")
    return "test_data.jsonl"

def run_fine_tuning(data_file):
    """Run the fine-tuning process."""
    print("\n=== Running Fine-Tuning ===")
    cmd = f"python risk_fine_tuner.py --training-data {data_file} --output test_output"
    print(f"Command: {cmd}")
    
    # Note: This would actually run the fine-tuning
    # For now, we'll just show what would be run
    print("(Fine-tuning would start here)")
    
    # The fine-tuner creates: test_output/unified_model_with_categories.pkl
    return "test_output/unified_model_with_categories.pkl"

def test_inference(pickle_path):
    """Test the inference with the pickle file."""
    print("\n=== Testing Inference ===")
    
    # Test single text analysis
    test_texts = [
        "Critical vulnerability found in authentication system allowing privilege escalation",
        "Employee John Smith, SSN 555-12-3456, has access to production database"
    ]
    
    for text in test_texts:
        cmd = f'python risk_inference.py --model {pickle_path} --text "{text}"'
        print(f"\nTest command: {cmd}")
        print("(Inference would run here)")
    
    # Test batch analysis
    with open("test_batch.txt", "w") as f:
        f.write("Firewall rules not properly configured\n")
        f.write("Customer credit card: 4111-1111-1111-1111\n")
        f.write("Backup systems have not been tested in 12 months\n")
    
    cmd = f"python risk_inference.py --model {pickle_path} --file test_batch.txt --output test_results.json"
    print(f"\nBatch test command: {cmd}")
    print("(Batch inference would run here)")

def main():
    """Main test function."""
    print("=== Integration Test: risk_fine_tuner.py ↔ risk_inference.py ===")
    
    # Step 1: Create test data
    data_file = create_test_data()
    
    # Step 2: Run fine-tuning (or simulate it)
    pickle_path = run_fine_tuning(data_file)
    
    # Step 3: Test inference
    test_inference(pickle_path)
    
    print("\n=== Integration Flow ===")
    print("1. risk_fine_tuner.py processes training data (JSONL, raw Excel/CSV)")
    print("2. Creates unified_model_with_categories.pkl containing:")
    print("   - model_path: Path to fine-tuned model")
    print("   - unified: True")
    print("   - l2: L2 category definitions")
    print("   - macro_risks: Macro risk definitions")
    print("   - pii_protection_categories: PC0/PC1/PC3 definitions")
    print("   - pii_types: Common PII type definitions")
    print("3. risk_inference.py loads the pickle and performs analysis")
    print("4. Automatically detects if text is risk or PII and analyzes accordingly")
    
    print("\n✅ Integration test setup complete!")
    print("\nTo run the actual test:")
    print("1. python risk_fine_tuner.py --training-data test_data.jsonl --output test_output")
    print("2. python risk_inference.py --model test_output/unified_model_with_categories.pkl --text \"Your text\"")

if __name__ == "__main__":
    main() 