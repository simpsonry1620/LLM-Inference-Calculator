"""
Command-line interface for the LLM Infrastructure Scaling Calculator.
"""
import argparse
import json
import logging
import os
from .calculator import LLMScalingCalculator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LLM Infrastructure Scaling Calculator"
    )
    
    parser.add_argument(
        "--model-size", 
        type=float, 
        required=True,
        help="Model size in billions of parameters (e.g., 7 for 7B)"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--seq-length", 
        type=int, 
        default=2048,
        help="Sequence length for training"
    )
    
    parser.add_argument(
        "--tokens-to-train", 
        type=float, 
        default=300,
        help="Tokens to train on in billions (e.g., 300 for 300B tokens)"
    )
    
    parser.add_argument(
        "--gpu-memory", 
        type=float, 
        default=80,
        help="GPU memory in GB (default: 80GB for A100)"
    )
    
    parser.add_argument(
        "--gpu-flops", 
        type=float, 
        default=312,
        help="GPU TFLOPS (default: 312 for A100)"
    )
    
    parser.add_argument(
        "--gpu-cost", 
        type=float, 
        default=1.5,
        help="GPU cost per hour in USD (default: $1.5)"
    )
    
    parser.add_argument(
        "--format", 
        choices=["text", "json"], 
        default="text",
        help="Output format (text or json)"
    )
    
    return parser.parse_args()


def format_results_text(results, args):
    """Format results as human-readable text."""
    summary = results["summary"]
    memory = results["memory_requirements"]
    compute = results["compute_requirements"]
    
    output = []
    output.append("=" * 60)
    output.append(f"LLM INFRASTRUCTURE SCALING CALCULATOR RESULTS")
    output.append("=" * 60)
    output.append(f"Model size: {summary['model_size_billions']:.1f}B parameters")
    output.append(f"Batch size: {args.batch_size}")
    output.append(f"Sequence length: {args.seq_length}")
    output.append(f"Training tokens: {args.tokens_to_train:.1f}B")
    output.append("-" * 60)
    
    output.append("\nMEMORY REQUIREMENTS:")
    output.append(f"Model parameters: {memory['model_size_gb']:.2f} GB")
    output.append(f"Optimizer states: {memory['optimizer_size_gb']:.2f} GB")
    output.append(f"Activations: {memory['activation_size_gb']:.2f} GB")
    output.append(f"Total memory: {memory['total_memory_gb']:.2f} GB")
    
    output.append("\nCOMPUTE REQUIREMENTS:")
    output.append(f"Total compute: {compute['total_petaflops']:.2f} petaFLOPS")
    output.append(f"GPU hours: {compute['gpu_hours']:,.0f}")
    
    output.append("\nRESOURCE SUMMARY:")
    output.append(f"GPUs needed: {memory['gpus_needed']}")
    output.append(f"Training time: {compute['training_days']:.1f} days")
    output.append(f"Estimated cost: ${compute['training_cost_usd']:,.2f}")
    output.append(f"Scaling efficiency: {compute['scaling_efficiency']:.2f}")
    
    output.append("=" * 60)
    
    return "\n".join(output)


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    # Setup logging
    log_dir = "logging"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, "cli_calculations.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler() # Also print logs to console if needed
        ]
    )
    
    logging.info("Starting CLI calculation.")
    logging.info(f"Arguments: {vars(args)}")
    
    # Convert to raw values
    model_size_params = args.model_size * 1e9
    tokens_to_train = args.tokens_to_train * 1e9
    gpu_flops = args.gpu_flops * 1e12
    
    # Initialize calculator
    calculator = LLMScalingCalculator(
        gpu_memory_gb=args.gpu_memory,
        gpu_flops=gpu_flops,
        gpu_cost_per_hour=args.gpu_cost
    )
    
    # Calculate resources
    results = calculator.estimate_resources(
        model_size_params=model_size_params,
        batch_size=args.batch_size,
        sequence_length=args.seq_length,
        tokens_to_train=tokens_to_train
    )
    
    # Output results
    if args.format == "json":
        results_json = json.dumps(results, indent=2)
        print(results_json)
        logging.info(f"Calculation results (JSON): {results_json}")
    else:
        results_text = format_results_text(results, args)
        print(results_text)
        logging.info(f"Calculation results (Text):\\n{results_text}")
    
    print(f"\nCalculation results have been logged to: logs/calculations/")
    logging.info("Finished CLI calculation.")


if __name__ == "__main__":
    main() 