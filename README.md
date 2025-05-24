# Foundation-Sec-8B Risk & PII Analyzer

A suite of tools for fine-tuning and using the Foundation-Sec-8B model for security risk categorization and PII detection.

## Overview

This project provides tools to:

1. **Fine-tune** the Foundation-Sec-8B model on a unified taxonomy for both risk categorization and PII classification
2. **Analyze** security risks and automatically categorize them
3. **Detect PII** in text and classify it into protection categories (PC0, PC1, PC3)

The tools can process any text and automatically determine whether to perform risk analysis or PII detection.

## Components

The project consists of two main components:

1. **risk_fine_tuner.py**: For fine-tuning a unified model that handles both risk taxonomy and PII classification
2. **risk_inference.py**: For using the fine-tuned model to analyze text (automatically detecting the appropriate task)

## Risk Taxonomy & PII Classification

The system uses:

### Risk Categorization
- 12 Macro Risk Categories
- Multiple Thematic Risks for each category

For example:
- **Category 8**: Manage IT Vulnerabilities & Patching
  - Scanning Completeness
  - Patching Completeness
  - S-SDLC drafts
  - Vulnerability assessment and risk treatment

### PII Protection Categories
- **PC0**: Public information with no confidentiality requirements
- **PC1**: Internal information with basic confidentiality requirements  
- **PC3**: Confidential information with high protection requirements

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- tqdm
- pandas

For fine-tuning, a GPU with at least 16GB VRAM is recommended.

## Installation

1. Clone this repository:
```
git clone [repository-url]
cd foundation-sec-8b-risk-analyzer
```

2. Install dependencies:
```
pip install -r requirements.txt
```

## Usage

### Fine-Tuning

Train a unified model that can handle both risk categorization and PII detection:

```
python risk_fine_tuner.py --training_data path/to/training_data --output_dir fine_tuning_output
```

The training data can include:
- Risk categorization examples
- PII classification examples
- A mix of both

The system will automatically detect the type of each example and train a unified model.

Data can be provided in various formats:
- CSV files
- JSON/JSONL files
- Excel files
- A directory containing multiple data files

After fine-tuning completes, a pickle file will be created with the model and all categories for inference.

### Inference

The inference script automatically detects whether to perform risk analysis or PII detection:

```
python risk_inference.py --model path/to/unified_model_with_categories.pkl --text "Your text to analyze"
```

To process multiple texts from a file:

```
python risk_inference.py --model path/to/unified_model_with_categories.pkl --file path/to/texts.txt --output results.json
```

## Data Format

### Input Formats for Fine-Tuning

The system supports multiple input formats for training data:

#### CSV Format for Risk Categorization
```csv
Risk Finding Text,Macro Risk Category,Thematic Risk 1,Thematic Risk 2
"Missing encryption for sensitive data in transit","5. Protect Data","Encryption (At Rest, Use, Transit)"
```

#### CSV Format for PII Detection
```csv
Text with potential PII,PC Category,PII Type 1,PII Type 2
"Customer John Smith contacted us about order #12345","PC1","Name","Customer ID"
"Our public documentation is available at docs.example.com","PC0"
"Patient medical records contain diagnoses and SSNs","PC3","Health","SSN"
```

#### Mixed CSV Format
You can also provide a single CSV file with mixed data types. The system will automatically detect the type of each row:
```csv
Text,Category,Type1,Type2
"Missing encryption for sensitive data in transit","5. Protect Data","Encryption (At Rest, Use, Transit)"
"Customer John Smith contacted us about order #12345","PC1","Name","Customer ID"
```

#### JSON/JSONL Format
Each example should indicate its type:
```json
{
  "type": "risk",
  "text": "Missing encryption for sensitive data in transit",
  "macro_risk": "5. Protect Data",
  "risk_themes": ["Encryption (At Rest, Use, Transit)"]
}
```

```json
{
  "type": "pii",
  "text": "Customer John Smith contacted us about order #12345",
  "pc_category": "PC1",
  "pii_types": ["Name", "Customer ID"]
}
```

### Output Format

The inference results include a "type" field indicating whether it's a risk or PII analysis:

#### Risk Categorization Results
```json
{
  "type": "risk",
  "success": true,
  "macro_risk": "5. Protect Data",
  "risk_themes": ["Encryption (At Rest, Use, Transit)"],
  "input_text": "Missing encryption for sensitive data in transit"
}
```

#### PII Detection Results
```json
{
  "type": "pii",
  "success": true,
  "pc_category": "PC1",
  "pii_types": ["Name", "Customer ID"],
  "input_text": "Customer John Smith contacted us about order #12345"
}
```

## License

Please refer to the license of the Foundation-Sec-8B model from Cisco.

## Acknowledgments

- This project uses the Foundation-Sec-8B model developed by Cisco's Security AI team
- Fine-tuning is implemented using Hugging Face Transformers and PEFT