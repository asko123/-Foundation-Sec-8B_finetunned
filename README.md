# Foundation-Sec-8B Fine-Tuner with H100 Checkpointing

Unified fine-tuning system for security risk categorization and PII detection with robust checkpointing for H100 GPUs.

## Features

- **Risk Categorization**: L2 categories and macro risks mapping
- **PII Detection**: PC0/PC1/PC3 classification with "highest sensitivity wins" rule  
- **DDL Analysis**: Database schema PII detection with ambiguous classification
- **H100 Checkpointing**: Auto-resume from interruptions with optimized batch sizes
- **Time-Limited Training**: Built for 4-day GPU time limits

## Training with Checkpoints

### Start New Training
```bash
# Using shell script (recommended)
./run_gradient_fixed.sh --data ./training_data --output ./h100_output

# Using Python directly
python risk_fine_tuner_gradient_fixed.py --training-data ./training_data --output ./h100_output
```

### Resume from Latest Checkpoint (Auto-detect)
```bash
./run_gradient_fixed.sh --data ./training_data --output ./h100_output
# Will auto-detect checkpoints and prompt to resume
```

### Resume from Specific Checkpoint
```bash
./run_gradient_fixed.sh --data ./training_data --output ./h100_output --resume ./h100_output/checkpoints/checkpoint-1000
```

### Force Fresh Start (Ignore Checkpoints)
```bash
./run_gradient_fixed.sh --data ./training_data --output ./h100_output --no-auto-resume
```

## H100 Configurations

| VRAM | Batch Size | Grad Accumulation | Save Steps | Precision |
|------|------------|-------------------|------------|-----------|
| 80GB (H100) | 4 | 8 | 25 | bf16 |
| 40GB (H100/A100) | 2 | 16 | 50 | bf16 |
| 20GB (RTX 4090) | 1 | 32 | 100 | fp16 |
| <20GB | 1 | 64 | 200 | fp16 |

## Checkpoint Management

- **Automatic saves**: Every 25-200 steps (based on VRAM)
- **Time-based saves**: Every 30 minutes
- **Interruption handling**: Graceful Ctrl+C with state saving
- **Error recovery**: Auto-save on training exceptions
- **Multiple retention**: Keeps 10 checkpoints for safety

## File Structure
```
output/
├── checkpoints/checkpoint-{step}/    # Training checkpoints
├── final_model/                      # Final trained model
├── fixed_model.pkl                   # Inference package
└── training_state.json               # Resume metadata
```

## Running Inference

### Load and Analyze Text
```python
from risk_inference import load_inference_model, analyze_text

# Load trained model
model, tokenizer, unified, categories = load_inference_model("./output/fixed_model.pkl")

# Analyze security risk finding
result = analyze_text(model, tokenizer, unified, categories, "Password policy lacks complexity requirements")
print(f"L2 Category: {result['l2_category']}")
print(f"Macro Risks: {result['macro_risks']}")

# Analyze PII text
result = analyze_text(model, tokenizer, unified, categories, "John Smith, SSN: 123-45-6789")
print(f"PC Category: {result['pc_category']}")
print(f"PII Types: {result['pii_types']}")
```

### Command Line Inference
```bash
# Analyze single text
python risk_inference.py --model ./output/fixed_model.pkl --text "Database lacks encryption"

# Analyze file
python risk_inference.py --model ./output/fixed_model.pkl --file findings.txt --output results.json

# Supported file formats: txt, json, jsonl, csv, xlsx
```

## Training Data Formats

### Risk Examples
```json
{
  "type": "risk",
  "text": "Database lacks encryption at rest",
  "l2_category": "5. Protect Data", 
  "macro_risks": ["Encryption (At Rest, Use, Transit)"]
}
```

### PII Examples  
```json
{
  "type": "pii",
  "text": "Customer John Smith, email: john@company.com",
  "pc_category": "PC1",
  "pii_types": ["Name", "Email"]
}
```

### DDL Examples
```json
{
  "type": "ddl", 
  "ddl_statement": "CREATE TABLE users (id INT, email VARCHAR(255), ssn CHAR(11))",
  "analysis_result": {
    "overall_classification": "PC3",
    "detected_pii_types": ["Email", "SSN"],
    "requires_human_review": false
  }
}
```

## Installation

```bash
pip install -r requirements.txt
```

## Categories

### L2 Risk Categories
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

### PII Protection Categories
- **PC0**: Public information (no confidentiality requirements)
- **PC1**: Internal information (basic confidentiality requirements)  
- **PC3**: Confidential information (high protection requirements)

## Transfer Between Machines

```bash
# Package training state
tar -czf training.tar.gz ./output/

# On new machine
tar -xzf training.tar.gz
./run_gradient_fixed.sh --data ./training_data --output ./output
# Auto-detects and resumes from checkpoint
```

## **How Checkpoints Work - Summary:**

**🔄 AUTOMATIC OPERATION:**
- **Training saves every 25 steps** (H100) automatically
- **Time-based backup every 30 minutes** 
- **Just run the same command** to resume - it detects checkpoints automatically

**📁 WHAT GETS SAVED:**
- Model weights (LoRA adapters)
- Optimizer state (momentum, learning rates)
- Training progress (current step, epoch)
- Processed training data (no reprocessing needed)

**🚀 RESUME PROCESS:**
1. Script scans `./output/checkpoints/` folder
2. Finds highest numbered checkpoint (e.g., `checkpoint-875`)
3. Prompts: `Resume from this checkpoint? [Y/n]:`
4. Press **Enter** (default YES) → continues from step 876
5. Training picks up **exactly where it stopped**

**💾 4-DAY WORKFLOW:**
```bash
# Day 1-3: Start training
./run_gradient_fixed.sh --data ./training_data --output ./h100_session

# Day 4: Copy entire folder to new machine
tar -czf session.tar.gz ./h100_session/

# New machine: Same command auto-resumes
./run_gradient_fixed.sh --data ./training_data --output ./h100_session
```

**🎯 KEY BENEFIT:**
**Zero manual work** - just run the same command and it automatically continues where you left off. Perfect for time-limited GPU rentals!
