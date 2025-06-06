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

# Create and activate virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

There are three main ways to use this system:

1. **Enhanced Data Processing & Fine-Tuning (Recommended)**:
```bash
# Process raw data and optionally start fine-tuning
python run_enhanced_fine_tuner.py --folder /path/to/raw/data --run-fine-tuning

# This will:
# 1. Process all Excel/CSV files in the folder
# 2. Generate training examples
# 3. Start fine-tuning if --run-fine-tuning is specified
```

2. **Direct Fine-Tuning** (if you have pre-formatted data):
```bash
python risk_fine_tuner.py --training-data your_data.jsonl --output my_model
```

3. **Inference** (after fine-tuning):
```bash
python risk_inference.py --model my_model/unified_model_with_categories.pkl --text "Your text here"
```

## 📊 Data Processing

### Input Data Support

The system now supports various input formats and field names:

#### Title Fields:
- Finding_Title, Title, FINDING_TITLE, TITLE
- Name, NAME, Summary, SUMMARY
- Issue_Title, ISSUE, Observation_Title
- Vulnerability_Title, Vuln_Title, Heading, Subject

#### Description Fields:
- Finding_Description, Description, DESCRIPTION
- Details, DETAILS, Content, CONTENT
- Issue_Description, Detail, Observation
- Observation_Details, Vulnerability_Description
- Finding_Details, Notes, Comments, Body

#### Category Fields:
- L2, Category, Type, Classification
- Risk_Category, Finding_Category
- Vulnerability_Type, Vuln_Type, Risk_Type
- Issue_Type, Severity, Risk_Classification
- Security_Classification

#### Risk Fields:
- macro_risks, Risks, Risk_Types, Categories
- Risk_Tags, Tags, Risk_Classification
- Threat_Types, Threat_Categories
- Impact_Types, Vulnerability_Tags
- Security_Tags, Risk_Areas, Risk_Domains

### Output Files

The system generates several output files:

```
output_directory/
├── auto_generated_training_data.jsonl  # Processed training examples
├── extraction_summary.json             # Processing statistics
├── unified_model_with_categories.pkl   # Fine-tuned model package
└── fine_tuned_model/                  # Model files
```

### Example Output Statistics

```
Extracted 1000 examples:
- Complete examples: 600    (with both L2 and risks)
- With L2 only: 100        (missing risks)
- With risks only: 50      (missing L2)
- Partial examples: 250    (useful but incomplete)
```

## 🔧 Advanced Usage

### Processing Options

```bash
# Process data with custom options
python run_enhanced_fine_tuner.py \
  --folder /path/to/data \
  --output custom_output \
  --run-fine-tuning

# Process and save to specific directory
python run_enhanced_fine_tuner.py \
  --folder /path/to/data \
  --output my_output \
  --training-data training_data
```

### Linux Environment Setup

If you encounter library issues on Linux:

```bash
# Fix environment issues
./fix_threading.sh

# Run scripts with fixes
./run_python_fixed.sh run_enhanced_fine_tuner.py --folder /path/to/data
```

## 🐛 Troubleshooting

### Common Issues

1. **"No training examples could be generated"**
   - Check your input data format
   - Verify column names match supported fields
   - Ensure files contain meaningful text content
   - Check the extraction_summary.json for details

2. **Memory Issues**
   - Data is processed in batches of 5000 rows
   - Reduce batch size if needed in the code
   - Use virtual environment with clean dependencies

3. **Excel/CSV Reading Errors**
   - Verify file permissions
   - Check file encoding (UTF-8, latin1, etc.)
   - Ensure files aren't corrupted or locked

### Debug Output

The system now provides focused error reporting:
- File processing status
- Critical errors only
- Processing statistics
- Extraction summary

## 📚 Risk Categories

The system uses standardized risk categories:

**L2 Categories:**
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

Each L2 category has associated **Macro Risks** that represent specific vulnerabilities or weaknesses within that category.

**PII Protection Categories:**
- PC0: Public information (no confidentiality requirements)
- PC1: Internal information (basic confidentiality)
- PC3: Confidential information (high protection requirements)

## 📝 License

[Your License Here]

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📧 Support

For issues or questions, please open a GitHub issue or contact [your contact info].
