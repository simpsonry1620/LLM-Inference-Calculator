#!/usr/bin/env python3
"""
Model Comparison Example

This script demonstrates how to use the predefined model functionality to
compare different LLM architectures without specifying all parameters manually.
"""

import sys
import os
import json
from typing import List, Dict, Any

# Add the parent directory to the path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.advanced_calculator import AdvancedCalculator


def print_section(title):
    """Print a formatted section title"""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)


def print_model_info(model_info):
    """Print basic information about a model"""
    print(f"\n----- {model_info['name']} ({model_info['family']}) -----")
    print(f"Parameters:         {model_info['parameter_count']:.1f}B")
    print(f"Hidden dimensions:  {model_info['hidden_dimensions']}")
    print(f"FF dimensions:      {model_info['feedforward_dimensions']}")
    print(f"Number of layers:   {model_info['num_layers']}")
    print(f"Default seq length: {model_info['default_seq_length']}")
    print(f"Description:        {model_info['description']}")


def print_comparison_table(results, metric, column_width=15):
    """Print a comparison table for a specific metric"""
    # Get all unique keys from all dictionaries
    headers = ["Model"]
    
    # Extract all metrics from the first model as column headers
    if results:
        first_model = list(results.values())[0]
        metric_dict = first_model
        for path in metric.split('.'):
            if path in metric_dict:
                metric_dict = metric_dict[path]
            else:
                print(f"Metric path {metric} not found")
                return
        
        if isinstance(metric_dict, dict):
            headers.extend(metric_dict.keys())
        else:
            headers.append("Value")
    
    # Print headers
    header_line = "| "
    for header in headers:
        header_str = str(header)
        if len(header_str) > column_width - 2:
            header_str = header_str[:column_width-5] + "..."
        header_line += header_str.ljust(column_width) + " | "
    print(header_line)
    
    # Print separator
    separator = "|" + "-" * (len(header_line) - 2) + "|"
    print(separator)
    
    # Print rows
    for model_name, result in sorted(results.items()):
        row = f"| {model_name.ljust(column_width)} | "
        
        # Navigate to the specific metric in the nested dictionary
        metric_value = result
        for path in metric.split('.'):
            if isinstance(metric_value, dict) and path in metric_value:
                metric_value = metric_value[path]
            else:
                metric_value = "N/A"
                break
        
        if isinstance(metric_value, dict):
            # If the metric is a dictionary, print each value as a column
            for header in headers[1:]:
                if header in metric_value:
                    value = metric_value[header]
                    if isinstance(value, (int, float)):
                        value_str = f"{value:.2f}"
                    else:
                        value_str = str(value)
                    row += value_str.ljust(column_width) + " | "
                else:
                    row += "N/A".ljust(column_width) + " | "
        else:
            # Otherwise, print the value directly
            if isinstance(metric_value, (int, float)):
                value_str = f"{metric_value:.2f}"
            else:
                value_str = str(metric_value)
            row += value_str.ljust(column_width) + " | "
        
        print(row)


def format_vram(vram_gb):
    """Format VRAM in a readable way with appropriate units"""
    if vram_gb < 1:
        return f"{vram_gb * 1024:.1f} MB"
    elif vram_gb < 10:
        return f"{vram_gb:.2f} GB"
    else:
        return f"{vram_gb:.1f} GB"


def format_flops(flops):
    """Format FLOPs in a readable way with appropriate units"""
    if flops < 1e9:
        return f"{flops / 1e6:.2f} MFLOPs"
    elif flops < 1e12:
        return f"{flops / 1e9:.2f} GFLOPs"
    else:
        return f"{flops / 1e12:.2f} TFLOPs"


def main():
    """Main function to demonstrate model comparison"""
    calculator = AdvancedCalculator()
    
    print_section("Available Model Families")
    model_families = calculator.get_model_families()
    print(f"Available model families: {', '.join(model_families)}")
    
    print_section("Available Models")
    all_models = calculator.get_available_models()
    print(f"Total available models: {len(all_models)}")
    
    for family in model_families:
        models_in_family = calculator.get_models_by_family(family)
        model_names = [model["name"] for model in models_in_family]
        print(f"\n{family}: {', '.join(model_names)}")
    
    # Select specific models to compare
    models_to_compare = [
        "phi-2",
        "phi-3-mini",
        "mistral-7b",
        "llama2-7b",
        "llama3-8b",
        "llama2-70b",
        "llama3-70b"
    ]
    
    print_section("Model Details")
    for model_name in models_to_compare:
        model_config = calculator.get_model_config(model_name)
        if model_config:
            print_model_info(model_config)
    
    # Analyze models with consistent settings
    print_section("Model Analysis")
    
    # Use consistent settings across all models
    seq_length = 8192
    batch_size = 1
    precision = "fp16"
    gpu_tflops = 312.0  # A100
    efficiency = 0.3
    
    # Store analysis results
    results = {}
    for model_name in models_to_compare:
        print(f"\nAnalyzing {model_name}...")
        try:
            analysis = calculator.analyze_model_by_name(
                model_name, 
                sequence_length=seq_length,
                batch_size=batch_size,
                precision=precision,
                gpu_tflops=gpu_tflops,
                efficiency_factor=efficiency
            )
            results[model_name] = analysis
            
            # Print key metrics
            print(f"Model parameters: {analysis['model_info']['parameter_count']:.1f}B")
            print(f"Model VRAM: {format_vram(analysis['vram']['model_weights'])}")
            print(f"KV Cache VRAM: {format_vram(analysis['vram']['kv_cache'])}")
            print(f"Prefill latency: {analysis['performance']['prefill_latency']:.4f} seconds")
            print(f"Generation speed: {analysis['performance']['tokens_per_second']:.2f} tokens/sec")
            
        except ValueError as e:
            print(f"Error analyzing {model_name}: {e}")
    
    # Compare models across different metrics
    print_section("Model Comparisons")
    
    print("\nModel Size Comparison (VRAM)")
    print_comparison_table(results, "vram")
    
    print("\nInference Speed Comparison (tokens/sec on different GPUs)")
    print_comparison_table(results, "performance.throughput_by_gpu")
    
    print("\nLatency Comparison")
    latency_data = {}
    for model_name, analysis in results.items():
        latency_data[model_name] = {
            "Prefill (s)": analysis["performance"]["prefill_latency"],
            "Per Token (ms)": analysis["performance"]["token_latency"] * 1000,
            "1K Tokens (s)": analysis["performance"]["time_for_1000_tokens"]
        }
    
    # Print custom latency table
    print("\n| Model           | Prefill (s)     | Per Token (ms)  | 1K Tokens (s)   |")
    print("|" + "-" * 78 + "|")
    for model_name, metrics in sorted(latency_data.items()):
        line = f"| {model_name.ljust(15)} | "
        line += f"{metrics['Prefill (s)']:.4f}".ljust(15) + " | "
        line += f"{metrics['Per Token (ms)']:.4f}".ljust(15) + " | "
        line += f"{metrics['1K Tokens (s)']:.4f}".ljust(15) + " | "
        print(line)
    
    # Export data to JSON for external tools
    with open("model_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults exported to model_comparison_results.json")
    
    # Clean up
    calculator.clear_history()


if __name__ == "__main__":
    main() 