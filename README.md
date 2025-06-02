# Foundation-Sec-8B Fine-Tuning for Risk Analysis and PII Detection

This repository contains tools for fine-tuning the Foundation-Sec-8B model to perform two critical security tasks:
1. **Security Risk Categorization** - Classify security findings into 12 standardized macro risk categories and their specific themes
2. **PII Detection and Classification** - Identify personally identifiable information and classify it into protection categories (PC0/PC1/PC3)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Foundation-Sec-8B_finetunned

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

1. **Fine-tune the model with your data:**
```bash
# Using pre-formatted training data (JSONL)
python risk_fine_tuner.py --training-data training_data.jsonl --output my_model

# Using raw Excel/CSV files (automatic processing)
python risk_fine_tuner.py --training-data /path/to/excel/folder --output my_model
```

2. **Use the fine-tuned model for inference:**
```bash
# Analyze a single text
python risk_inference.py --model my_model/unified_model_with_categories.pkl --text "Critical SQL injection vulnerability found in login system"

# Batch analysis from file
python risk_inference.py --model my_model/unified_model_with_categories.pkl --file texts.txt --output results.json
```

## 📊 Data Formats

### Training Data Formats

The fine-tuner accepts multiple data formats:

#### 1. Pre-formatted JSONL (Recommended for labeled data)
Each line should be a JSON object with:
- **For risk examples:**
```json
{
  "type": "risk",
  "text": "The firewall configuration allows unrestricted access from external IPs",
  "macro_risk": "7. Manage Infrastructure",
  "risk_themes": ["Network Segmentation", "Configuration Management"]
}
```

- **For PII examples:**
```json
{
  "type": "pii",
  "text": "Customer John Doe, email: john@example.com, SSN: 123-45-6789",
  "pc_category": "PC3",
  "pii_types": ["Name", "Email", "SSN"]
}
```

#### 2. Raw Excel/CSV Files (Automatic processing)
Simply point to a folder containing Excel (.xlsx, .xls) or CSV files:
```bash
python risk_fine_tuner.py --training-data /path/to/raw/data/folder --output my_model
```

The system will:
- Extract text from all sheets and rows
- Automatically categorize content using 100+ security keywords
- Detect PII using pattern matching
- Generate training examples with confidence scores

## 🏗️ Architecture

### Components

1. **risk_fine_tuner.py** - Main fine-tuning script
   - Loads and processes training data
   - Performs LoRA fine-tuning for efficiency
   - Creates inference package (pickle file)

2. **risk_fine_tuner_enhanced.py** - Raw data processing
   - Extracts text from Excel/CSV files
   - Analyzes content for risk categories
   - Detects PII patterns
   - Generates training examples

3. **risk_inference.py** - Inference script
   - Loads fine-tuned model from pickle
   - Automatically detects if text is risk or PII
   - Returns structured JSON results

4. **run_enhanced_fine_tuner.py** - User-friendly wrapper
   - Simplified interface for processing raw data
   - Progress tracking and error handling

## 🔧 Advanced Usage

### Processing Raw Data

```bash
# Process a folder of Excel files with the enhanced script
python run_enhanced_fine_tuner.py /path/to/excel/files

# This creates:
# - extracted_data/training_data.jsonl (processed examples)
# - extracted_data/extraction_summary.json (statistics)
```

### Linux Environment Setup

If you encounter library issues on Linux:

```bash
# Use the provided fix scripts
./fix_threading.sh
./run_python_fixed.sh risk_fine_tuner.py --training-data data.jsonl
```

### Custom Categories

The system uses standardized risk categories defined in the scripts:

**12 Macro Risk Categories:**
1. Operating Model & Risk Management
2. Develop and Acquire Software and Systems
3. Manage & Demise IT Assets
4. Manage Data
5. Protect Data
6. Identity & Access Management
7. Manage Infrastructure
8. Manage IT Vulnerabilities & Patching
9. Manage Technology Capacity & Resources
10. Monitor & Respond to Technology Incidents
11. Monitor and Respond to Security Incidents
12. Manage Business Continuity and Disaster Recovery

**PII Protection Categories:**
- PC0: Public information (no confidentiality requirements)
- PC1: Internal information (basic confidentiality)
- PC3: Confidential information (high protection requirements)

## 📁 Output Files

After fine-tuning, you'll find:

```
output_directory/
├── unified_model_with_categories.pkl  # Main inference package
├── final_model/                       # Fine-tuned model files
├── unified_train.jsonl               # Formatted training data
├── unified_eval.jsonl                # Evaluation data
└── checkpoints/                      # Training checkpoints
```

## 🧪 Testing

Run the integration test to verify everything works:

```bash
python test_integration.py
```

This creates test data and shows the complete workflow.

## 💡 Examples

### Example 1: Analyze a security finding
```bash
python risk_inference.py \
  --model my_model/unified_model_with_categories.pkl \
  --text "Database backup encryption keys stored in plaintext configuration file"
```

Expected output:
```
Type: Security Risk
Macro Risk: 5. Protect Data
Risk Themes: Encryption (At Rest, Use, Transit), Secrets Management
```

### Example 2: Detect PII
```bash
python risk_inference.py \
  --model my_model/unified_model_with_categories.pkl \
  --text "Employee record: Jane Smith, DOB: 01/15/1985, Salary: $85,000"
```

Expected output:
```
Type: PII
Protection Category: PC3
PII Types: Name, DOB, Financial
```

## 🐛 Troubleshooting

### "No valid training examples found"
- Ensure your data is in the correct format
- For raw Excel files, check they contain text data
- Verify file permissions

### Memory issues during fine-tuning
- Reduce batch size in the script
- Use gradient accumulation
- Consider using a smaller dataset for initial testing

### Linux library errors
- Use the provided fix_threading.sh script
- Set environment variables: `export OPENBLAS_NUM_THREADS=1`

## 📝 License

[Your License Here]

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📧 Support

For issues or questions, please open a GitHub issue or contact [your contact info].
