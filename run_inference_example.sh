#!/bin/bash

# Example script to run inference
# Replace 'your_model.pkl' with the actual path to your trained model

MODEL_PATH="./model_output/fixed_model.pkl"

echo "=== Risk & PII Inference Examples ==="
echo ""

# Example 1: Single text analysis
echo "1. Analyzing single security risk text..."
python risk_inference.py --model "$MODEL_PATH" --text "SQL injection vulnerability found in login authentication allowing unauthorized database access and potential data exfiltration"

echo ""
echo "2. Analyzing single PII text..."
python risk_inference.py --model "$MODEL_PATH" --text "Customer record: John Smith, SSN: 123-45-6789, Email: john.smith@company.com, Phone: 555-123-4567"

echo ""
echo "3. Analyzing file with multiple texts..."
python risk_inference.py --model "$MODEL_PATH" --file sample_texts.txt --output analysis_results.json

echo ""
echo "4. Analyzing JSONL file..."
python risk_inference.py --model "$MODEL_PATH" --file sample_data.jsonl

echo ""
echo "=== Analysis Complete ==="
