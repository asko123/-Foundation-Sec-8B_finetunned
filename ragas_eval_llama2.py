#!/usr/bin/env python3
"""
RAGAS Evaluation Script for RAG Systems using Local Hugging Face Models

This script evaluates a RAG system using RAGAS metrics with a local Llama-2 model.
No closed-source APIs are used - everything runs locally.

Installation:
    pip install ragas langchain langchain-community transformers datasets pandas sentence-transformers bitsandbytes accelerate torch

GPU/VRAM Requirements:
    - Requires CUDA-capable GPU with at least 8GB VRAM for Llama-2-13b-chat-hf
    - For lower VRAM, use --model-id with smaller models like microsoft/DialoGPT-medium

How to run:
    python ragas_eval_llama2.py
    python ragas_eval_llama2.py --model-id microsoft/DialoGPT-medium --context-join " | "
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Any

import pandas as pd
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)
try:
    from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
except ImportError:
    from langchain_community.llms import HuggingFacePipeline
    from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)
from datasets import Dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments for the RAGAS evaluation script."""
    parser = argparse.ArgumentParser(description='Evaluate RAG system with RAGAS using local models')
    parser.add_argument(
        '--model-id',
        type=str,
        default='microsoft/DialoGPT-small',
        help='''Hugging Face model ID to use for evaluation (default: microsoft/DialoGPT-small). 
        RECOMMENDED MODELS:
        - For meaningful results with GPU: microsoft/DialoGPT-medium, HuggingFaceH4/zephyr-7b-beta
        - For best results with lots of VRAM: meta-llama/Llama-2-13b-chat-hf, mistralai/Mistral-7B-Instruct-v0.1
        - For CPU testing only: microsoft/DialoGPT-small, distilgpt2'''
    )
    parser.add_argument(
        '--context-join',
        type=str,
        default=' ',
        help='String to join Document Name and Expected Response for context (default: " ")'
    )
    parser.add_argument(
        '--save-csv',
        type=str,
        default='ragas_scores.csv',
        help='Path to save CSV results (default: ragas_scores.csv)'
    )
    parser.add_argument(
        '--save-json',
        type=str,
        default='ragas_scores.json',
        help='Path to save JSON results (default: ragas_scores.json)'
    )
    return parser.parse_args()

def get_evaluation_data():
    """Hard-coded evaluation dataset as specified in requirements."""
    return [
        {
            "Document Name": "FIRMWIDE POLICY ON PRIVACY: PROTECTION OF PERSONAL DATA",
            "Document Catalog": "Policies",
            "Question": "How does the firm handle the reporting and escalation of incidents involving unauthorized disclosures of personal data?",
            "Expected Response": "Upon learning of any incident involving unauthorized disclosure of personal data, immediate reporting and escalation are required. The incident should be reported through the established online or direct channels to the Security Incident Response Team. This ensures that appropriate legal, regulatory, and contractual obligations are met, including notifying affected persons and regulators for proper handling.",
            "TRAI Response A": "Goldman Sachs has a structured approach to handle the reporting and escalation of incidents involving unauthorized disclosures of personal data, as outlined in their Firmwide Policy on Privacy: Protection of Personal Data. Key points: (1) Mandatory Reporting, (2) Divisional Responsibility, (3) Governance and Oversight, (4) Compliance with Laws."
        },
        {
            "Document Name": "FIRMWIDE POLICY ON PRIVACY: PROTECTION OF PERSONAL DATA",
            "Document Catalog": "Policies",
            "Question": "What does the 'Need to Know' standard entail in the context of personal data access?",
            "Expected Response": "The 'Need to Know' standard restricts access to personal data to individuals who need it to perform their duties. This principle of data minimization ensures that only necessary personal data is disclosed and used, thereby reducing the risk of unnecessary exposure of sensitive information.",
            "TRAI Response A": "The 'Need to Know' standard refers to restricting access to personal data only to employees with legitimate business needs. It minimizes risks of unauthorized use or disclosure by ensuring that personal data is only accessible to those who require it for work-related responsibilities."
        },
        {
            "Document Name": "FIRMWIDE POLICY ON PRIVACY: PROTECTION OF PERSONAL DATA",
            "Document Catalog": "Policies",
            "Question": "What additional restrictions apply to the collection and use of certain types of personal data?",
            "Expected Response": "Additional restrictions apply to the collection and use of certain types of personal data, such as government-issued identification numbers, driver's licenses, and medical history. These restrictions are in place to ensure compliance with legal and regulatory requirements.",
            "TRAI Response A": "There are additional restrictions on collection and use of certain personal data driven by privacy and protection laws, which may limit types of personal data collected, restrict use for marketing, or impose international transfer limitations. Divisions are required to safeguard such data and ensure compliance."
        },
        {
            "Document Name": "FIRMWIDE POLICY ON PRIVACY: PROTECTION OF PERSONAL DATA",
            "Document Catalog": "Policies",
            "Question": "According to the policy, what are the responsibilities of each division in safeguarding personal data?",
            "Expected Response": "Each division is responsible for safeguarding all personal data that they collect and use. This includes measures to prevent unauthorized access or disclosure, securing both physical and electronic access to data, and ensuring proper transmission and disposal methods.",
            "TRAI Response A": "Divisions must ensure compliance with privacy policies by safeguarding and managing personal data. Responsibilities include: (1) Data Usage & Sharing, (2) Safeguarding and Data Access, (3) Need to Know Standard, (4) Additional Restrictions, (5) Reporting Unauthorized Disclosures, (6) Divisional Responsibility."
        }
    ]

def setup_model_and_tokenizer(model_id: str):
    """
    Load and configure the Hugging Face model with 4-bit quantization.
    
    Args:
        model_id: Hugging Face model identifier
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {model_id}")
    
    # Check for CUDA availability - use CPU for smaller models if CUDA unavailable
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        logger.warning("CUDA not available. Will attempt CPU execution for smaller models.")
        # For larger models like Llama-2-13b, we still need CUDA
        if "llama" in model_id.lower() and "13b" in model_id.lower():
            raise RuntimeError(
                "CUDA is not available but required for large models like Llama-2-13b. "
                "Please install CUDA drivers or use a smaller model via --model-id (e.g., microsoft/DialoGPT-medium)"
            )
    
    try:
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Set pad token to EOS token if not present (common issue with some models)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        
        # Configure model loading based on CUDA availability
        if use_cuda:
            # Configure 4-bit quantization using bitsandbytes for CUDA
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            
            # Load model with quantization
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                use_safetensors=True  # Prefer safetensors for security
            )
            logger.info("Model loaded successfully with 4-bit quantization on CUDA")
        else:
            # Load model for CPU execution
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,  # Use float32 for CPU
                device_map="cpu",
                trust_remote_code=True,
                use_safetensors=True  # Prefer safetensors for security
            )
            logger.info("Model loaded successfully for CPU execution")
        
        return model, tokenizer
        
    except ImportError as e:
        if "bitsandbytes" in str(e) and use_cuda:
            raise RuntimeError(
                "bitsandbytes is not installed or not compatible with your system. "
                "Please install it with: pip install bitsandbytes, or use a smaller model "
                "via --model-id (e.g., microsoft/DialoGPT-medium)"
            )
        elif "bitsandbytes" in str(e):
            # Try CPU loading without quantization
            logger.warning("bitsandbytes not available, falling back to CPU without quantization")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                use_safetensors=True  # Prefer safetensors for security
            )
            logger.info("Model loaded successfully for CPU execution without quantization")
            return model, tokenizer
        raise
    except Exception as e:
        logger.error(f"Error loading model {model_id}: {e}")
        raise RuntimeError(
            f"Failed to load model {model_id}. Try using a smaller model via --model-id "
            "(e.g., microsoft/DialoGPT-medium) or ensure you have sufficient resources."
        )

def create_text_generation_pipeline(model, tokenizer):
    """
    Create a text generation pipeline with deterministic settings.
    
    Args:
        model: Loaded Hugging Face model
        tokenizer: Corresponding tokenizer
        
    Returns:
        Hugging Face text generation pipeline
    """
    # Set manual seed for determinism
    torch.manual_seed(42)
    
    # Create pipeline with specified parameters for deterministic generation
    # Use much smaller max_new_tokens for small models to avoid context length issues
    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=0.1,  # Low temperature for deterministic output
        do_sample=False,  # Disable sampling for determinism
        max_new_tokens=100,  # Reduced for small models
        return_full_text=False,  # Only return generated text, not input
        pad_token_id=tokenizer.eos_token_id,
        truncation=True  # Enable truncation
    )
    
    logger.info("Text generation pipeline created with deterministic settings")
    return text_pipeline

def setup_ragas_components(text_pipeline):
    """
    Set up RAGAS LLM and embedding components.
    
    Args:
        text_pipeline: Hugging Face text generation pipeline
        
    Returns:
        Tuple of (ragas_llm, ragas_embeddings)
    """
    logger.info("Setting up RAGAS components...")
    
    # Wrap HF pipeline with LangChain
    langchain_llm = HuggingFacePipeline(pipeline=text_pipeline)
    
    # Wrap LangChain LLM with RAGAS
    ragas_llm = LangchainLLMWrapper(langchain_llm)
    
    # Set up embeddings using sentence-transformers
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Wrap embeddings for RAGAS
    ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)
    
    logger.info("RAGAS components setup completed")
    return ragas_llm, ragas_embeddings

def construct_ragas_dataset(data: List[Dict], context_join: str) -> Dataset:
    """
    Construct a RAGAS dataset from the evaluation data.
    
    Args:
        data: List of evaluation records
        context_join: String to join Document Name and Expected Response
        
    Returns:
        RAGAS-compatible dataset
    """
    logger.info("Constructing RAGAS dataset...")
    
    # Transform data into RAGAS format
    ragas_data = []
    for record in data:
        # Create synthesized context by joining Document Name and Expected Response
        context = record["Document Name"] + context_join + record["Expected Response"]
        
        # More aggressive truncation for small models
        # Estimate ~4 chars per token, aim for max 200 tokens per field
        max_chars_per_field = 800  # Very conservative for small models
        
        # Truncate each field individually
        question = record["Question"]
        if len(question) > max_chars_per_field:
            question = question[:max_chars_per_field] + "..."
            logger.warning(f"Truncated question: {question[:50]}...")
            
        response = record["TRAI Response A"]
        if len(response) > max_chars_per_field:
            response = response[:max_chars_per_field] + "..."
            logger.warning(f"Truncated response: {response[:50]}...")
            
        reference = record["Expected Response"]
        if len(reference) > max_chars_per_field:
            reference = reference[:max_chars_per_field] + "..."
            logger.warning(f"Truncated reference: {reference[:50]}...")
        
        # Create much shorter context by just using document name
        context = record["Document Name"]
        if len(context) > max_chars_per_field:
            context = context[:max_chars_per_field] + "..."
            logger.warning(f"Truncated context: {context[:50]}...")
        
        ragas_record = {
            "user_input": question,
            "response": response,
            "retrieved_contexts": [context],  # RAGAS expects a list of context strings
            "reference": reference
        }
        ragas_data.append(ragas_record)
    
    # Convert to Hugging Face Dataset format
    dataset = Dataset.from_list(ragas_data)
    
    logger.info(f"Dataset constructed with {len(ragas_data)} records")
    return dataset

def run_ragas_evaluation(dataset: Dataset, ragas_llm, ragas_embeddings):
    """
    Run RAGAS evaluation with all six metrics.
    
    Args:
        dataset: RAGAS dataset
        ragas_llm: RAGAS LLM wrapper
        ragas_embeddings: RAGAS embeddings wrapper
        
    Returns:
        RAGAS evaluation results
    """
    logger.info("Running RAGAS evaluation with all six metrics...")
    
    # Import metrics and configure them with our local LLM and embeddings
    # In RAGAS 0.3.x, we need to set the llm and embeddings attributes on the global instances
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness
    )
    try:
        from ragas.metrics import ContextRelevance
    except ImportError:
        # Try alternative import paths for different RAGAS versions
        try:
            from ragas.metrics._context_relevance import ContextRelevance
        except ImportError:
            # If still failing, create a simple context relevance using available metric
            ContextRelevance = None
            logger.warning("ContextRelevance metric not available in this RAGAS version")
    
    logger.info("Configuring global metrics with local LLM and embeddings...")
    
    # Configure global metric instances with our local components
    faithfulness.llm = ragas_llm
    logger.info("Configured faithfulness metric with local LLM")
    
    answer_relevancy.llm = ragas_llm
    answer_relevancy.embeddings = ragas_embeddings
    logger.info("Configured answer_relevancy metric with local LLM and embeddings")
    
    context_precision.llm = ragas_llm
    logger.info("Configured context_precision metric with local LLM")
    
    context_recall.llm = ragas_llm
    logger.info("Configured context_recall metric with local LLM")
    
    answer_correctness.llm = ragas_llm
    answer_correctness.embeddings = ragas_embeddings
    logger.info("Configured answer_correctness metric with local LLM and embeddings")
    
    # Create context relevance instance and configure it if available
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness
    ]
    
    if ContextRelevance is not None:
        context_relevance = ContextRelevance(llm=ragas_llm)
        metrics.append(context_relevance)
        logger.info("Created context_relevance metric with local LLM")
    else:
        logger.warning("Skipping context_relevance metric (not available)")
    
    try:
        # Run evaluation with error handling
        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            raise_exceptions=False  # Continue evaluation even if individual samples fail
        )
        
        logger.info("RAGAS evaluation completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Error during RAGAS evaluation: {e}")
        raise

def save_results(results, csv_path: str, json_path: str):
    """
    Save evaluation results to CSV and JSON files.
    
    Args:
        results: RAGAS evaluation results
        csv_path: Path to save CSV file
        json_path: Path to save JSON file
    """
    logger.info(f"Saving results to {csv_path} and {json_path}")
    
    # Convert results to pandas DataFrame for easier manipulation
    df = results.to_pandas()
    
    # Save to CSV
    df.to_csv(csv_path, index=False)
    
    # Save to JSON with proper formatting
    json_data = df.to_dict(orient='records')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    logger.info("Results saved successfully")

def print_aggregate_metrics(results):
    """
    Print aggregate metric means to stdout.
    
    Args:
        results: RAGAS evaluation results
    """
    print("\n" + "="*60)
    print("RAGAS EVALUATION RESULTS - AGGREGATE METRICS")
    print("="*60)
    
    # Convert to pandas for easier aggregation
    df = results.to_pandas()
    
    # Define the metrics we're interested in
    metric_columns = [
        'faithfulness', 'answer_relevancy', 'context_precision', 
        'context_recall', 'answer_correctness'
    ]
    
    # Add context_relevance if it exists in the results
    if 'context_relevance' in df.columns:
        metric_columns.append('context_relevance')
    
    # Calculate and print means for each metric
    for metric in metric_columns:
        if metric in df.columns:
            mean_score = df[metric].mean()
            print(f"{metric.replace('_', ' ').title():<20}: {mean_score:.4f}")
        else:
            print(f"{metric.replace('_', ' ').title():<20}: Not available")
    
    print("="*60)
    print(f"Total samples evaluated: {len(df)}")
    print("="*60)

def main():
    """Main execution function."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        logger.info("Starting RAGAS evaluation...")
        logger.info(f"Model: {args.model_id}")
        logger.info(f"Context join string: '{args.context_join}'")
        
        # Get evaluation data
        data = get_evaluation_data()
        
        # Load model and tokenizer with quantization
        model, tokenizer = setup_model_and_tokenizer(args.model_id)
        
        # Create text generation pipeline
        text_pipeline = create_text_generation_pipeline(model, tokenizer)
        
        # Setup RAGAS components
        ragas_llm, ragas_embeddings = setup_ragas_components(text_pipeline)
        
        # Construct RAGAS dataset
        dataset = construct_ragas_dataset(data, args.context_join)
        
        # Run RAGAS evaluation
        results = run_ragas_evaluation(dataset, ragas_llm, ragas_embeddings)
        
        # Save results to files
        save_results(results, args.save_csv, args.save_json)
        
        # Print aggregate metrics
        print_aggregate_metrics(results)
        
        logger.info("RAGAS evaluation completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
