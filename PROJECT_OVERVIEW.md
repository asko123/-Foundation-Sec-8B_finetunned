# Foundation-Sec-8B Project Overview

## 📁 **Project Structure**

```
Foundation-Sec-8B_finetunned/
├── 🎯 CORE SCRIPTS
│   ├── risk_fine_tuner_enhanced.py     # Enhanced data processor
│   ├── run_enhanced_fine_tuner.py      # Main entry point
│   ├── risk_fine_tuner.py              # Model fine-tuning
│   └── risk_inference.py               # Inference script
│
├── 📊 DATA & OUTPUT
│   ├── training_data/                  # Generated training data
│   └── fine_tuned_model/              # Fine-tuned model files
│
├── 📚 DOCUMENTATION
│   ├── README.md                       # Complete user guide
│   └── PROJECT_OVERVIEW.md             # This overview
│
├── 🔧 SUPPORT SCRIPTS
│   ├── run_python_fixed.sh             # Linux environment wrapper
│   └── fix_threading.sh                # Linux environment setup
│
└── 📦 CONFIGURATION
    ├── requirements.txt                # Python dependencies
    └── risk_fine_tuner_constants.py    # Risk categories & constants
```

## 🚀 **Quick Start Guide**

### **1. Setup Environment**
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **2. Process & Fine-tune (Recommended Flow)**
```bash
# Process data and start fine-tuning in one go
python run_enhanced_fine_tuner.py \
  --folder /path/to/data \
  --run-fine-tuning

# Or process only
python run_enhanced_fine_tuner.py \
  --folder /path/to/data
```

### **3. Run Inference**
```bash
python risk_inference.py \
  --model fine_tuned_model/unified_model_with_categories.pkl \
  --text "your text here"
```

## 📋 **Script Functions**

### **Main Scripts**

| Script | Purpose | Input | Output |
|--------|---------|-------|---------|
| `run_enhanced_fine_tuner.py` | **Main entry point** | Excel/CSV files | Training data & optionally fine-tuned model |
| `risk_fine_tuner_enhanced.py` | **Data processor** | Raw data | Processed training examples |
| `risk_fine_tuner.py` | **Model trainer** | Training data | Fine-tuned model |
| `risk_inference.py` | **Predictor** | Text input | Risk/PII predictions |

### **Support Scripts**

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `run_python_fixed.sh` | Linux fixes | When running on Linux |
| `fix_threading.sh` | Environment setup | Once per Linux session |

## 🔄 **Processing Flow**

1. **Data Loading**
   - Reads Excel (.xlsx, .xls) and CSV files
   - Supports multiple sheets
   - Handles various encodings

2. **Field Mapping**
   - Maps various column names to standard fields
   - Supports multiple naming conventions
   - Preserves metadata

3. **Content Processing**
   - Extracts meaningful text
   - Cleans and normalizes content
   - Handles partial data

4. **Example Generation**
   - Creates training examples
   - Assigns categories and risks
   - Preserves context

5. **Output Generation**
   - Saves processed examples
   - Generates statistics
   - Creates summary report

## 📊 **Output Files**

```
output_directory/
├── auto_generated_training_data.jsonl  # Training examples
├── extraction_summary.json             # Processing stats
├── unified_model_with_categories.pkl   # Fine-tuned model
└── fine_tuned_model/                  # Model files
```

## 🔍 **Supported Fields**

### Title Variations
- Finding_Title, Title, Name
- Issue_Title, Summary
- Vulnerability_Title, etc.

### Description Variations
- Finding_Description, Description
- Details, Content, Notes
- Observation, etc.

### Category Fields
- L2, Category, Type
- Classification, Risk_Category
- Vulnerability_Type, etc.

### Risk Fields
- macro_risks, Risks
- Risk_Types, Categories
- Risk_Tags, etc.

## ⚠️ **Common Issues**

1. **No Examples Generated**
   - Check input data format
   - Verify column names
   - Check file permissions

2. **Memory Issues**
   - Uses batch processing (5000 rows)
   - Clean virtual environment
   - Monitor memory usage

3. **Linux Issues**
   - Use provided fix scripts
   - Check environment setup
   - Verify permissions

## 🎯 **Typical Workflow**

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Process your data**: `python run_enhanced_fine_tuner.py --folder /your/data`
3. **Review results**: Check `training_data/extraction_summary.json`
4. **Optional fine-tuning**: `python risk_fine_tuner.py --training-data training_data/auto_generated_training_data.jsonl`
5. **Optional inference**: `python risk_inference.py --model model.pkl --input "test text"`

## 🧪 **Testing**

### **Test Commands**
```bash
# Lightweight tests (tests core functionality without loading the full model)
python test_lightweight.py

# 8-bit quantization tests (validates quantization implementation)
python test_8bit_quantization.py

# Full integration test (requires ~15GB RAM for the full model)
python test_integration.py
```

### **Test Results**
- **Lightweight tests**: 4/4 tests pass ✅
- **Quantization tests**: 5/5 tests pass ✅
- **Integration test**: Limited by memory constraints in some environments

Make sure you're in the repo directory and have the virtual environment activated with `source .venv/bin/activate` before running tests.

## 🔧 **Linux Users**

Replace `python` with `./run_python_fixed.sh` in all commands:
```bash
./run_python_fixed.sh run_enhanced_fine_tuner.py --folder /your/data
./run_python_fixed.sh risk_fine_tuner.py --training-data training_data/auto_generated_training_data.jsonl
./run_python_fixed.sh test_lightweight.py
./run_python_fixed.sh test_8bit_quantization.py
```

## 📊 **What This System Does**

✅ **Automatically processes** Excel (.xlsx, .xls) and CSV files  
✅ **Extracts security content** from spreadsheets  
✅ **Categorizes risks** into 12 standardized categories  
✅ **Detects PII** and classifies protection levels  
✅ **Generates training data** for machine learning  
✅ **Fine-tunes models** for your specific data  
✅ **Handles Linux environment issues** automatically  

**Perfect for**: Security audits, risk assessments, compliance reports, vulnerability scans  