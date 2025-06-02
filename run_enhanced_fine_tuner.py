#!/usr/bin/env python3
"""
Enhanced Fine-Tuner Runner - Automatically processes raw Excel/CSV files and prepares training data.

This script:
1. Scans a folder for Excel/CSV files
2. Extracts content and analyzes it for risk categories and PII
3. Generates training examples automatically
4. Prepares data for fine-tuning

Usage:
    python run_enhanced_fine_tuner.py --folder /path/to/raw/data/folder

The script will automatically:
- Find all Excel (.xlsx, .xls) and CSV files in the folder
- Extract text content from each file
- Analyze content to identify risk categories and PII classifications
- Generate training examples with confidence scores
- Save processed training data for fine-tuning
"""

import os
import sys
import argparse
import traceback
from risk_fine_tuner_enhanced import process_folder_for_training_data

def main():
    """Main function to run the enhanced fine-tuner."""
    parser = argparse.ArgumentParser(
        description="Enhanced Fine-Tuner - Process raw Excel/CSV files and extract training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process a folder containing Excel/CSV files
    python run_enhanced_fine_tuner.py --folder ./data_files
    
    # Process files and specify output directory
    python run_enhanced_fine_tuner.py --folder ./data_files --output ./training_output
    
    # Process with custom confidence threshold
    python run_enhanced_fine_tuner.py --folder ./data_files --confidence 0.2
        """
    )
    
    parser.add_argument(
        "--folder", 
        required=True, 
        help="Path to folder containing raw Excel/CSV files to process"
    )
    
    parser.add_argument(
        "--output", 
        default="training_data", 
        help="Directory to save extracted training data (default: training_data)"
    )
    
    parser.add_argument(
        "--confidence", 
        type=float, 
        default=0.1, 
        help="Minimum confidence threshold for including examples (default: 0.1)"
    )
    
    args = parser.parse_args()
    
    try:
        print("=" * 60)
        print("🚀 ENHANCED RISK & PII FINE-TUNER")
        print("=" * 60)
        print(f"📁 Input folder: {args.folder}")
        print(f"📄 Output directory: {args.output}")
        print(f"🎯 Confidence threshold: {args.confidence}")
        print()
        
        # Check if input folder exists
        if not os.path.exists(args.folder):
            print(f"❌ Error: Input folder '{args.folder}' does not exist!")
            return 1
        
        if not os.path.isdir(args.folder):
            print(f"❌ Error: '{args.folder}' is not a directory!")
            return 1
        
        # Process the folder and extract training data
        print("🔍 Starting raw data processing...")
        training_file = process_folder_for_training_data(args.folder, args.output)
        
        print()
        print("✅ PROCESSING COMPLETE!")
        print("=" * 60)
        print(f"📋 Training data file: {training_file}")
        print(f"📊 Summary report: {os.path.join(args.output, 'extraction_summary.json')}")
        print()
        print("🎯 NEXT STEPS:")
        print("1. Review the extracted training examples")
        print("2. Run fine-tuning with the generated data:")
        print(f"   python risk_fine_tuner.py --training-data {training_file}")
        print()
        print("📁 Generated files:")
        
        # List generated files
        output_files = []
        for file in os.listdir(args.output):
            file_path = os.path.join(args.output, file)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                output_files.append(f"   • {file} ({size_mb:.2f} MB)")
        
        for file_info in output_files:
            print(file_info)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        print("\n📋 Full error details:")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 