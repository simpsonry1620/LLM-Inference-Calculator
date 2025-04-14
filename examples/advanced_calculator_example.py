#!/usr/bin/env python3
"""
Advanced Calculator Example

This script demonstrates how to use the advanced calculator to estimate
computational requirements for different model sizes and configurations.
"""

import sys
import os

# Add the parent directory to the path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.advanced_calculator import AdvancedCalculator


def print_section(title):
    """Print a formatted section title"""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)


def print_model_stats(name, params, hidden_dim, ff_dim, num_layers, seq_len, batch=1):
    """Print basic model information"""
    print(f"\n----- {name} Model -----")
    print(f"Parameters:         {params / 1e9:.1f}B")
    print(f"Hidden dimensions:  {hidden_dim}")
    print(f"FF dimensions:      {ff_dim}")
    print(f"Number of layers:   {num_layers}")
    print(f"Sequence length:    {seq_len}")
    print(f"Batch size:         {batch}")


def analyze_model(calculator, name, hidden_dim, ff_dim, num_layers, vocab_size, 
                 seq_len=2048, batch_size=1, gpu_tflops=312.0):
    """Run a comprehensive analysis for a specific model configuration"""
    # Estimate parameters
    attention_params = 4 * hidden_dim * hidden_dim * num_layers  # QKV + output projections
    ff_params = 2 * hidden_dim * ff_dim * num_layers  # Up & down projections
    embedding_params = vocab_size * hidden_dim  # Token embeddings
    layernorm_params = 4 * hidden_dim * num_layers + 2 * hidden_dim  # Layer norms + final LN
    total_params = attention_params + ff_params + embedding_params + layernorm_params
    
    print_model_stats(name, total_params, hidden_dim, ff_dim, num_layers, seq_len, batch_size)
    
    # FLOPs calculations
    flops_attention = calculator.calculate_flops_attention(batch_size, seq_len, hidden_dim)
    flops_feedforward = calculator.calculate_flops_feedforward(batch_size, seq_len, hidden_dim, ff_dim)
    flops_prefill = calculator.calculate_flops_prefill(batch_size, seq_len, hidden_dim, ff_dim, num_layers)
    
    # Calculate per-token FLOPs (for a single new token)
    flops_per_token = (
        # For single token: seq_len=1 for new token's attention to all previous tokens
        calculator.calculate_flops_attention(batch_size, 1, hidden_dim) +
        calculator.calculate_flops_feedforward(batch_size, 1, hidden_dim, ff_dim)
    ) * num_layers
    
    # VRAM calculations
    model_vram_fp16 = calculator.calculate_model_vram(
        hidden_dim, ff_dim, num_layers, vocab_size, "fp16"
    )
    model_vram_fp32 = calculator.calculate_model_vram(
        hidden_dim, ff_dim, num_layers, vocab_size, "fp32"
    )
    
    kv_cache_vram = calculator.calculate_kv_cache_vram(
        batch_size, seq_len, hidden_dim, num_layers
    )
    
    total_vram = calculator.calculate_total_vram(
        batch_size, seq_len, hidden_dim, ff_dim, num_layers, vocab_size
    )
    
    # Throughput and latency
    tokens_per_second = calculator.estimate_inference_throughput(
        flops_per_token, gpu_tflops, 0.3  # Using 30% efficiency
    )
    
    prefill_latency = calculator.estimate_prefill_latency(
        flops_prefill, gpu_tflops, 0.3
    )
    
    token_latency = calculator.estimate_token_generation_latency(
        flops_per_token, gpu_tflops, 0.3
    )
    
    # Output results
    print("\n----- FLOPs Calculations -----")
    print(f"Attention FLOPs:      {flops_attention:,}")
    print(f"Feedforward FLOPs:    {flops_feedforward:,}")
    print(f"Total Prefill FLOPs:  {flops_prefill:,}")
    print(f"FLOPs per token:      {flops_per_token:,}")
    
    print("\n----- VRAM Requirements -----")
    print(f"Model VRAM (FP16):    {model_vram_fp16:.2f} GB")
    print(f"Model VRAM (FP32):    {model_vram_fp32:.2f} GB")
    print(f"KV Cache VRAM:        {kv_cache_vram:.2f} GB")
    print(f"Total VRAM (FP16):    {total_vram:.2f} GB")
    
    print("\n----- Performance Estimates -----")
    print(f"Inference speed:      {tokens_per_second:.2f} tokens/sec")
    print(f"Prefill latency:      {prefill_latency:.4f} seconds")
    print(f"Per-token latency:    {token_latency:.4f} seconds")
    print(f"Time for 1000 tokens: {token_latency * 1000:.2f} seconds")


def main():
    """Main function to demonstrate the calculator"""
    calculator = AdvancedCalculator()
    
    print_section("Advanced Calculator Examples")
    
    # Model configurations
    # Format: name, hidden_dim, ff_dim, num_layers, vocab_size, seq_len
    models = [
        ("Small", 768, 3072, 12, 50257, 2048),
        ("Medium", 1024, 4096, 24, 50257, 4096),
        ("Large", 4096, 16384, 32, 120000, 8192),
        ("XL", 8192, 28672, 80, 128000, 8192)
    ]
    
    # Analyze each model
    for model in models:
        analyze_model(calculator, *model)
        print("\n" + "-" * 80)
    
    # GPU efficiency comparison
    print_section("GPU Efficiency Comparison (XL Model)")
    
    model = models[3]  # XL model
    hidden_dim, ff_dim, num_layers, vocab_size, seq_len = model[1:]
    
    print("\nTokens per second at different efficiency levels:")
    
    # Calculate per-token FLOPs for XL model
    flops_per_token = (
        (calculator.calculate_flops_attention(1, 1, hidden_dim) +
         calculator.calculate_flops_feedforward(1, 1, hidden_dim, ff_dim))
        * num_layers
    )
    
    # Compare different GPU configurations and efficiencies
    gpu_configs = [
        ("A100 (40GB)", 312.0),
        ("A100 (80GB)", 312.0),
        ("H100", 756.0),
        ("H200", 989.0)
    ]
    
    efficiency_levels = [0.1, 0.2, 0.3, 0.5, 0.8]
    
    print("\n{:<15} | {:<12} | {:<12} | {:<12} | {:<12} | {:<12}".format(
        "GPU", "10% Eff", "20% Eff", "30% Eff", "50% Eff", "80% Eff"
    ))
    print("-" * 85)
    
    for gpu_name, tflops in gpu_configs:
        results = []
        for eff in efficiency_levels:
            tokens_per_sec = calculator.estimate_inference_throughput(
                flops_per_token, tflops, eff
            )
            results.append(f"{tokens_per_sec:.1f}")
        
        print("{:<15} | {:<12} | {:<12} | {:<12} | {:<12} | {:<12}".format(
            gpu_name, *results
        ))
    
    # Clean up
    calculator.clear_history()


if __name__ == "__main__":
    main() 