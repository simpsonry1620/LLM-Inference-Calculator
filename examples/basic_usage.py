"""
Basic usage example for the LLM Infrastructure Scaling Calculator.
"""
import sys
import os

# Add parent directory to path to import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.calculator import LLMScalingCalculator


def main():
    """Demonstrate basic usage of the calculator."""
    # Initialize calculator with default settings (A100 GPU)
    calculator = LLMScalingCalculator()
    
    # Define model parameters
    model_sizes = [1e9, 7e9, 13e9, 70e9, 175e9]  # 1B, 7B, 13B, 70B, 175B
    batch_size = 32
    sequence_length = 2048
    tokens = 300e9  # 300B tokens
    
    print("LLM INFRASTRUCTURE SCALING ANALYSIS")
    print("=" * 80)
    print(f"{'Model Size':15} {'GPUs':8} {'Training Days':15} {'Cost (USD)':15}")
    print("-" * 80)
    
    # Calculate and print estimates for different model sizes
    for params in model_sizes:
        results = calculator.estimate_resources(
            model_size_params=params,
            batch_size=batch_size,
            sequence_length=sequence_length,
            tokens_to_train=tokens
        )
        
        summary = results["summary"]
        size_b = summary["model_size_billions"]
        gpus = summary["gpus_needed"]
        days = summary["training_days"]
        cost = summary["training_cost_usd"]
        
        print(f"{size_b:6.1f}B params {gpus:8d} {days:15.1f} ${cost:15,.2f}")
    
    print("=" * 80)
    print("Notes:")
    print("- These estimates are based on theoretical calculations")
    print("- GPU: NVIDIA A100 (80GB, 312 TFLOPS)")
    print("- Cost estimate: $1.5 per GPU hour")
    print("- Scaling efficiency is modeled as 2/sqrt(num_gpus) with max of 100%")


if __name__ == "__main__":
    main() 