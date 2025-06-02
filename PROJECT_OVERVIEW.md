# Foundation-Sec-8B Project Overview

## 📁 **Clean Repository Structure**

```
Foundation-Sec-8B_finetunned/
├── 🎯 CORE FUNCTIONALITY
│   ├── risk_fine_tuner_enhanced.py     # Enhanced raw data processor
│   ├── run_enhanced_fine_tuner.py      # Easy-to-use wrapper script
│   ├── risk_fine_tuner.py              # Fine-tuning script
│   └── risk_inference.py               # Model inference script
│
├── 📚 DOCUMENTATION
│   ├── README.md                       # Complete user guide
│   └── PROJECT_OVERVIEW.md             # This file - simple overview
│
├── 🔧 LINUX SUPPORT
│   ├── run_python_fixed.sh             # Python wrapper with environment fixes
│   └── fix_threading.sh                # Comprehensive Linux environment setup
│
└── 📦 DEPENDENCIES
    └── requirements.txt                 # Python package dependencies
```

## 🚀 **Quick Usage Guide**

### **Step 1: Process Raw Data**
```bash
# Basic usage
python run_enhanced_fine_tuner.py --folder /path/to/your/excel/files

# On Linux with environment issues
./run_python_fixed.sh run_enhanced_fine_tuner.py --folder /path/to/your/excel/files
```

### **Step 2: Fine-Tune Model (Optional)**
```bash
# Use the generated training data
python risk_fine_tuner.py --training-data training_data/auto_generated_training_data.jsonl

# On Linux
./run_python_fixed.sh risk_fine_tuner.py --training-data training_data/auto_generated_training_data.jsonl
```

### **Step 3: Run Inference (Optional)**
```bash
# Test the model
python risk_inference.py --model fine_tuned_model/model.pkl --input "your risk text"
```

## 📋 **File Descriptions**

### **Core Scripts**

| File | Purpose | When to Use |
|------|---------|-------------|
| `run_enhanced_fine_tuner.py` | **Main entry point** - Processes raw Excel/CSV files | Always start here with your raw data |
| `risk_fine_tuner_enhanced.py` | **Core processor** - Contains all the analysis logic | Used automatically by the wrapper |
| `risk_fine_tuner.py` | **Model training** - Fine-tunes the Foundation-Sec-8B model | After you have training data |
| `risk_inference.py` | **Model testing** - Runs predictions on new data | After fine-tuning is complete |

### **Linux Support**

| File | Purpose | When to Use |
|------|---------|-------------|
| `run_python_fixed.sh` | **Environment wrapper** - Handles library issues | Use instead of `python` on Linux |
| `fix_threading.sh` | **Environment setup** - Fixes threading and library paths | Run once per session on Linux |

### **Documentation**

| File | Purpose |
|------|---------|
| `README.md` | Complete user guide with examples and troubleshooting |
| `PROJECT_OVERVIEW.md` | This file - simple overview of the project |

## 🎯 **Typical Workflow**

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Process your data**: `python run_enhanced_fine_tuner.py --folder /your/data`
3. **Review results**: Check `training_data/extraction_summary.json`
4. **Optional fine-tuning**: `python risk_fine_tuner.py --training-data training_data/auto_generated_training_data.jsonl`
5. **Optional inference**: `python risk_inference.py --model model.pkl --input "test text"`

## 🔧 **Linux Users**

Replace `python` with `./run_python_fixed.sh` in all commands:
```bash
./run_python_fixed.sh run_enhanced_fine_tuner.py --folder /your/data
./run_python_fixed.sh risk_fine_tuner.py --training-data training_data/auto_generated_training_data.jsonl
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