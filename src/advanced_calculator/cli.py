import click
import json
from typing import Optional
from tabulate import tabulate
from src.advanced_calculator.main import AdvancedCalculator

@click.group()
def cli():
    """Advanced calculator for LLM computation requirements."""
    pass

@cli.command()
@click.option('--hidden-dim', type=int, required=True, help='Hidden dimension size')
@click.option('--ff-dim', type=int, required=True, help='Feedforward dimension size')
@click.option('--num-layers', type=int, required=True, help='Number of layers')
@click.option('--seq-len', type=int, required=True, help='Sequence length')
@click.option('--output-len', type=int, default=128, help='Output sequence length')
@click.option('--batch-size', type=int, default=1, help='Batch size')
@click.option('--vocab-size', type=int, default=50257, help='Vocabulary size')
@click.option('--precision', type=click.Choice(['fp16', 'fp32', 'bf16']), default='fp16', help='Model precision')
@click.option('--tflops', type=float, default=312.0, help='GPU TFLOPS (e.g., 312 for A100)')
@click.option('--efficiency', type=float, default=0.3, help='GPU efficiency factor')
@click.option('--format', type=click.Choice(['table', 'json']), default='table', help='Output format')
def calculate(hidden_dim, ff_dim, num_layers, seq_len, output_len, batch_size, vocab_size, precision, tflops, efficiency, format):
    """Calculate model requirements with custom parameters."""
    calculator = AdvancedCalculator()
    
    # FLOPs calculations
    flops_attention = calculator.calculate_flops_attention(batch_size, seq_len, hidden_dim)
    flops_feedforward = calculator.calculate_flops_feedforward(batch_size, seq_len, hidden_dim, ff_dim)
    flops_prefill = calculator.calculate_flops_prefill(batch_size, seq_len, hidden_dim, ff_dim, num_layers)
    
    # Calculate per-token FLOPs (for a single new token)
    flops_per_token = (
        calculator.calculate_flops_attention(batch_size, 1, hidden_dim) +
        calculator.calculate_flops_feedforward(batch_size, 1, hidden_dim, ff_dim)
    ) * num_layers
    
    # VRAM calculations
    model_vram = calculator.calculate_model_vram(hidden_dim, ff_dim, num_layers, vocab_size, precision)
    kv_cache_vram = calculator.calculate_kv_cache_vram(batch_size, seq_len, hidden_dim, num_layers, precision)
    total_vram = calculator.calculate_total_vram(batch_size, seq_len, hidden_dim, ff_dim, num_layers, vocab_size, precision)
    
    # Throughput and latency
    tokens_per_second = calculator.estimate_inference_throughput(flops_per_token, tflops, efficiency)
    prefill_latency = calculator.estimate_prefill_latency(flops_prefill, tflops, efficiency)
    token_latency = calculator.estimate_token_generation_latency(flops_per_token, tflops, efficiency)
    
    # Calculate TTFT and Total Request Time
    time_to_first_token = prefill_latency
    total_request_time = prefill_latency + (output_len * token_latency)
    time_for_1000_tokens = token_latency * 1000 # Keep this for comparison
    
    results = {
        "model_parameters": {
            "hidden_dimensions": hidden_dim,
            "feedforward_dimensions": ff_dim,
            "num_layers": num_layers,
            "sequence_length": seq_len,
            "output_sequence_length": output_len,
            "batch_size": batch_size,
            "vocab_size": vocab_size,
            "precision": precision
        },
        "compute_requirements": {
            "flops_attention": f"{flops_attention:.2e} FLOPs",
            "flops_feedforward": f"{flops_feedforward:.2e} FLOPs",
            "flops_prefill_total": f"{flops_prefill:.2e} FLOPs",
            "flops_per_token": f"{flops_per_token:.2e} FLOPs",
        },
        "memory_requirements": {
            "model_weights": f"{model_vram:.2f} GB",
            "kv_cache": f"{kv_cache_vram:.2f} GB",
            "total_vram": f"{total_vram:.2f} GB"
        },
        "performance_estimates": {
            "tokens_per_second": f"{tokens_per_second:.2f}",
            "prefill_latency": f"{prefill_latency:.4f} s",
            "token_latency": f"{token_latency:.4f} s",
            "time_to_first_token": f"{time_to_first_token:.4f} s",
            "total_request_time": f"{total_request_time:.2f} s",
            "time_for_1000_tokens": f"{time_for_1000_tokens:.2f} s"
        }
    }
    
    if format == 'json':
        click.echo(json.dumps(results, indent=2))
    else:
        # Create a more readable table output
        tables = []
        
        # Model Parameters Table
        model_params = [[k, v] for k, v in results["model_parameters"].items()]
        tables.append("\nModel Parameters:")
        tables.append(tabulate(model_params, headers=["Parameter", "Value"], tablefmt="grid"))
        
        # Compute Requirements Table
        compute_reqs = [[k, v] for k, v in results["compute_requirements"].items()]
        tables.append("\nCompute Requirements:")
        tables.append(tabulate(compute_reqs, headers=["Metric", "Value"], tablefmt="grid"))
        
        # Memory Requirements Table
        memory_reqs = [[k, v] for k, v in results["memory_requirements"].items()]
        tables.append("\nMemory Requirements:")
        tables.append(tabulate(memory_reqs, headers=["Component", "VRAM"], tablefmt="grid"))
        
        # Performance Estimates Table
        perf_ests = [[k, v] for k, v in results["performance_estimates"].items()]
        tables.append("\nPerformance Estimates:")
        tables.append(tabulate(perf_ests, headers=["Metric", "Value"], tablefmt="grid"))
        
        click.echo("\n".join(tables))

@cli.command()
@click.option('--model-name', required=True, help='Name of the predefined model')
@click.option('--seq-len', type=int, help='Input sequence length (uses model default if not specified)')
@click.option('--output-len', type=int, default=128, help='Desired output sequence length')
@click.option('--batch-size', type=int, default=1, help='Batch size')
@click.option('--precision', type=click.Choice(['fp16', 'fp32', 'bf16']), default='fp16', help='Model precision')
@click.option('--tflops', type=float, default=312.0, help='GPU TFLOPS (default: A100)')
@click.option('--efficiency', type=float, default=0.3, help='GPU efficiency factor')
@click.option('--format', type=click.Choice(['table', 'json']), default='table', help='Output format')
def analyze_model(model_name, seq_len, output_len, batch_size, precision, tflops, efficiency, format):
    """Analyze a predefined model."""
    calculator = AdvancedCalculator()
    
    try:
        results = calculator.analyze_model_by_name(
            model_name=model_name,
            sequence_length=seq_len,
            output_sequence_length=output_len,
            batch_size=batch_size,
            precision=precision,
            gpu_tflops=tflops,
            efficiency_factor=efficiency
        )
        
        if format == 'json':
            # Convert some values to strings to make JSON more readable
            results["flops"]["attention"] = f"{results['flops']['attention']:.2e}"
            results["flops"]["feedforward"] = f"{results['flops']['feedforward']:.2e}"
            results["flops"]["prefill_total"] = f"{results['flops']['prefill_total']:.2e}"
            results["flops"]["per_token"] = f"{results['flops']['per_token']:.2e}"
            
            # Format performance numbers for JSON
            results["performance"]["tokens_per_second"] = f"{results['performance']['tokens_per_second']:.2f}"
            results["performance"]["prefill_latency"] = f"{results['performance']['prefill_latency']:.4f} s"
            results["performance"]["token_latency"] = f"{results['performance']['token_latency']:.4f} s"
            results["performance"]["time_to_first_token"] = f"{results['performance']['time_to_first_token']:.4f} s"
            results["performance"]["total_request_time"] = f"{results['performance']['total_request_time']:.2f} s"
            results["performance"]["time_for_1000_tokens"] = f"{results['performance']['time_for_1000_tokens']:.2f} s"
            
            # Fix throughput values for JSON display
            for gpu, value in results["performance"]["throughput_by_gpu"].items():
                results["performance"]["throughput_by_gpu"][gpu] = f"{value:.2f} tokens/s"
            
            click.echo(json.dumps(results, indent=2))
        else:
            # Create a more readable table output
            tables = []
            
            # Model Info Table
            model_info = [
                ["Name", results["model_info"]["name"]],
                ["Family", results["model_info"]["family"]],
                ["Parameters", f"{results['model_info']['parameters_b']:.2f} B"],
                ["Hidden Dimensions", results["model_info"]["hidden_dimensions"]],
                ["FF Dimensions", results["model_info"]["feedforward_dimensions"]],
                ["Layers", results["model_info"]["num_layers"]],
                ["Description", results["model_info"]["description"]]
            ]
            tables.append("\nModel Information:")
            tables.append(tabulate(model_info, headers=["Attribute", "Value"], tablefmt="grid"))
            
            # Analysis Parameters Table
            analysis_params = [[k, v] for k, v in results["analysis_parameters"].items()]
            tables.append("\nAnalysis Parameters:")
            tables.append(tabulate(analysis_params, headers=["Parameter", "Value"], tablefmt="grid"))
            
            # FLOPS Table
            flops_table = [
                ["Attention", f"{results['flops']['attention']:.2e}"],
                ["Feedforward", f"{results['flops']['feedforward']:.2e}"],
                ["Prefill Total", f"{results['flops']['prefill_total']:.2e}"],
                ["Per Token", f"{results['flops']['per_token']:.2e}"]
            ]
            tables.append("\nFLOPs Requirements:")
            tables.append(tabulate(flops_table, headers=["Component", "FLOPs"], tablefmt="grid"))
            
            # VRAM Table
            vram_table = [
                ["Model Weights", f"{results['vram']['model_weights']:.2f} GB"],
                ["KV Cache", f"{results['vram']['kv_cache']:.2f} GB"],
                ["Total", f"{results['vram']['total']:.2f} GB"]
            ]
            tables.append("\nVRAM Requirements:")
            tables.append(tabulate(vram_table, headers=["Component", "VRAM"], tablefmt="grid"))
            
            # Performance Table
            perf_table = [
                ["Tokens Per Second", f"{results['performance']['tokens_per_second']:.2f}"],
                ["Prefill Latency", f"{results['performance']['prefill_latency']:.4f} s"],
                ["Token Generation Latency", f"{results['performance']['token_latency']:.4f} s"],
                ["Time to First Token (TTFT)", f"{results['performance']['time_to_first_token']:.4f} s"],
                ["Total Request Time", f"{results['performance']['total_request_time']:.2f} s"],
                ["Time for 1000 tokens", f"{results['performance']['time_for_1000_tokens']:.2f} s"]
            ]
            tables.append("\nPerformance Estimates:")
            tables.append(tabulate(perf_table, headers=["Metric", "Value"], tablefmt="grid"))
            
            # GPU Comparison Table
            gpu_comparison = [
                [gpu, f"{tps:.2f} tokens/s"] for gpu, tps in results["performance"]["throughput_by_gpu"].items()
            ]
            tables.append("\nPerformance Across GPUs:")
            tables.append(tabulate(gpu_comparison, headers=["GPU", "Tokens/s"], tablefmt="grid"))
            
            click.echo("\n".join(tables))
    
    except ValueError as e:
        click.echo(f"Error: {str(e)}", err=True)
        available_models = calculator.get_available_models()
        click.echo(f"\nAvailable models: {', '.join(available_models)}", err=True)

@cli.command()
def list_models():
    """List all available predefined models."""
    calculator = AdvancedCalculator()
    models = calculator.get_available_models()
    
    # Get model families
    families = calculator.get_model_families()
    
    click.echo("\nAvailable Model Families:")
    for family in families:
        click.echo(f"  - {family}")
    
    click.echo("\nAvailable Models:")
    table_data = []
    
    for model_name in models:
        model_config = calculator.get_model_config(model_name)
        table_data.append([
            model_name,
            model_config["family"],
            f"{model_config['parameters_b']:.2f} B",
            model_config["hidden_dimensions"],
            model_config["num_layers"],
            model_config.get("description", "")
        ])
    
    headers = ["Name", "Family", "Parameters", "Hidden Dim", "Layers", "Description"]
    click.echo(tabulate(table_data, headers=headers, tablefmt="grid"))

@cli.command()
def list_gpus():
    """List all available predefined GPUs."""
    calculator = AdvancedCalculator()
    gpus = calculator.get_available_gpus()
    
    # Get GPU families and generations
    families = calculator.get_gpu_families()
    generations = calculator.get_gpu_generations()
    
    click.echo("\nAvailable GPU Families:")
    for family in families:
        click.echo(f"  - {family}")
    
    click.echo("\nAvailable GPU Generations:")
    for generation in generations:
        click.echo(f"  - {generation}")
    
    click.echo("\nAvailable GPUs:")
    table_data = []
    
    for gpu_name in gpus:
        gpu_config = calculator.get_gpu_config(gpu_name)
        table_data.append([
            gpu_name,
            gpu_config["family"],
            gpu_config["generation"],
            f"{gpu_config['vram_gb']} GB",
            f"{gpu_config['fp16_tflops']:.1f}",
            f"{gpu_config['bandwidth_gb_per_sec']:.1f} GB/s",
            gpu_config.get("launch_year", "N/A")
        ])
    
    headers = ["Name", "Family", "Generation", "VRAM", "FP16 TFLOPs", "Bandwidth", "Year"]
    click.echo(tabulate(table_data, headers=headers, tablefmt="grid"))

@cli.command()
@click.option('--model-name', required=True, help='Name of the predefined model')
@click.option('--gpu-name', required=True, help='Name of the GPU to analyze on')
@click.option('--seq-len', type=int, help='Input sequence length (uses model default if not specified)')
@click.option('--output-len', type=int, default=128, help='Desired output sequence length')
@click.option('--batch-size', type=int, default=1, help='Batch size')
@click.option('--precision', type=click.Choice(['fp16', 'fp32', 'bf16', 'int8', 'int4']), default='fp16', help='Model precision')
@click.option('--efficiency', type=float, default=0.3, help='GPU efficiency factor')
@click.option('--format', type=click.Choice(['table', 'json']), default='table', help='Output format')
def analyze_model_on_gpu(model_name, gpu_name, seq_len, output_len, batch_size, precision, efficiency, format):
    """Analyze a model's performance on a specific GPU."""
    calculator = AdvancedCalculator()
    
    try:
        results = calculator.analyze_model_on_gpu(
            model_name=model_name,
            gpu_name=gpu_name,
            sequence_length=seq_len,
            output_sequence_length=output_len,
            batch_size=batch_size,
            precision=precision,
            efficiency_factor=efficiency
        )
        
        if format == 'json':
            click.echo(json.dumps(results, indent=2))
        else:
            # Create a more readable table output
            tables = []
            
            # Model Info Table
            model_info = [
                ["Name", results["model_info"]["name"]],
                ["Family", results["model_info"]["family"]],
                ["Parameters", f"{results['model_info']['parameters_b']:.2f} B"],
                ["Hidden Dimensions", results["model_info"]["hidden_dimensions"]],
                ["Layers", results["model_info"]["num_layers"]],
                ["Description", results["model_info"]["description"]]
            ]
            tables.append("\nModel Information:")
            tables.append(tabulate(model_info, headers=["Attribute", "Value"], tablefmt="grid"))
            
            # GPU Info Table
            gpu_info = [
                ["Name", results["gpu_info"]["name"]],
                ["Family", results["gpu_info"]["family"]],
                ["Generation", results["gpu_info"]["generation"]],
                ["VRAM", f"{results['gpu_info']['vram_gb']} GB"],
                [f"{precision.upper()} TFLOPs", results["gpu_info"].get(f"{precision}_tflops", "N/A")],
                ["Memory Bandwidth", f"{results['gpu_info']['bandwidth_gb_per_sec']} GB/s"],
                ["TDP", f"{results['gpu_info'].get('tdp_watts', 'N/A')} W"],
                ["Tensor Cores", "Yes" if results["gpu_info"].get("has_tensor_cores", False) else "No"],
                ["Year", results["gpu_info"].get("launch_year", "N/A")],
                ["Description", results["gpu_info"].get("description", "N/A")]
            ]
            tables.append("\nGPU Information:")
            tables.append(tabulate(gpu_info, headers=["Attribute", "Value"], tablefmt="grid"))
            
            # Analysis Parameters Table
            analysis_params = [
                ["Sequence Length", results["analysis_parameters"]["sequence_length"]],
                ["Batch Size", results["analysis_parameters"]["batch_size"]],
                ["Precision", results["analysis_parameters"]["precision"]],
                ["Precision Supported", "Yes" if results["analysis_parameters"]["precision_supported"] else "No"],
                ["Efficiency Factor", results["analysis_parameters"]["efficiency_factor"]]
            ]
            tables.append("\nAnalysis Parameters:")
            tables.append(tabulate(analysis_params, headers=["Parameter", "Value"], tablefmt="grid"))
            
            # Compatibility Table
            compatibility = [
                ["VRAM Required", f"{results['compatibility']['vram_required']:.2f} GB"],
                ["VRAM Available", f"{results['compatibility']['vram_available']:.2f} GB"],
                ["Model Fits", "Yes" if results['compatibility']['vram_fits'] else "No"],
                ["VRAM Headroom", f"{results['compatibility']['vram_headroom_gb']:.2f} GB"],
                ["Maximum Batch Size", results['compatibility']['maximum_batch_size']],
                ["Maximum Sequence Length", results['compatibility']['maximum_sequence_length']]
            ]
            tables.append("\nCompatibility Assessment:")
            tables.append(tabulate(compatibility, headers=["Metric", "Value"], tablefmt="grid"))
            
            # Performance Table
            perf_table = [
                ["Tokens Per Second", f"{results['performance']['tokens_per_second']:.2f}"],
                ["Prefill Latency", f"{results['performance']['prefill_latency']:.4f} s"],
                ["Token Generation Latency", f"{results['performance']['token_latency']:.4f} s"],
                ["Time to First Token (TTFT)", f"{results['performance']['time_to_first_token']:.4f} s"],
                ["Total Request Time", f"{results['performance']['total_request_time']:.2f} s"],
                ["Time for 1000 tokens", f"{results['performance']['time_for_1000_tokens']:.2f} s"]
            ]
            tables.append("\nPerformance Estimates:")
            tables.append(tabulate(perf_table, headers=["Metric", "Value"], tablefmt="grid"))
            
            click.echo("\n".join(tables))
    
    except ValueError as e:
        click.echo(f"Error: {str(e)}", err=True)
        available_models = calculator.get_available_models()
        available_gpus = calculator.get_available_gpus()
        click.echo(f"\nAvailable models: {', '.join(available_models)}", err=True)
        click.echo(f"\nAvailable GPUs: {', '.join(available_gpus)}", err=True)

@cli.command()
@click.option('--model-name', help='Name of the predefined model')
@click.option('--vram-gb', type=float, help='Minimum VRAM in GB for the model')
@click.option('--min-headroom-gb', type=float, default=2.0, help='Minimum VRAM headroom in GB')
@click.option('--format', type=click.Choice(['table', 'json']), default='table', help='Output format')
def recommend_gpus(model_name, vram_gb, min_headroom_gb, format):
    """Recommend GPUs for a specific model or VRAM requirement."""
    if not model_name and vram_gb is None:
        click.echo("Error: Either --model-name or --vram-gb must be specified", err=True)
        return
    
    calculator = AdvancedCalculator()
    
    try:
        model_name_or_vram = model_name if model_name else vram_gb
        recommended_gpus = calculator.get_recommended_gpus_for_model(
            model_name_or_vram=model_name_or_vram,
            min_vram_headroom_gb=min_headroom_gb
        )
        
        if format == 'json':
            click.echo(json.dumps(recommended_gpus, indent=2))
        else:
            if len(recommended_gpus) == 0:
                click.echo("No GPUs meet the requirements.")
                return
            
            header = "Recommended GPUs"
            if model_name:
                header = f"Recommended GPUs for {model_name}"
            elif vram_gb is not None:
                header = f"Recommended GPUs for {vram_gb:.1f} GB VRAM requirement"
            
            click.echo(f"\n{header}:")
            
            table_data = []
            for gpu in recommended_gpus:
                table_data.append([
                    gpu["name"],
                    gpu["family"],
                    gpu["generation"],
                    f"{gpu['vram_gb']} GB",
                    f"{gpu['fp16_tflops']:.1f}",
                    f"{gpu['bandwidth_gb_per_sec']:.1f} GB/s",
                    gpu.get("launch_year", "N/A"),
                    "Yes" if gpu.get("has_tensor_cores", False) else "No"
                ])
            
            headers = ["Name", "Family", "Generation", "VRAM", "FP16 TFLOPs", "Bandwidth", "Year", "Tensor Cores"]
            click.echo(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    except ValueError as e:
        click.echo(f"Error: {str(e)}", err=True)
        available_models = calculator.get_available_models()
        click.echo(f"\nAvailable models: {', '.join(available_models)}", err=True)

@cli.command()
@click.option('--min-vram', type=float, help='Minimum VRAM in GB')
@click.option('--precision', help='Required precision (e.g., fp16, bf16, fp8)')
@click.option('--family', help='Filter by GPU family')
@click.option('--generation', help='Filter by GPU generation')
@click.option('--format', type=click.Choice(['table', 'json']), default='table', help='Output format')
def filter_gpus(min_vram, precision, family, generation, format):
    """Filter GPUs by various criteria."""
    calculator = AdvancedCalculator()
    
    # Start with all GPUs
    filtered_gpus = calculator.get_available_gpus()
    gpu_configs = [calculator.get_gpu_config(gpu) for gpu in filtered_gpus]
    
    # Apply filters
    if min_vram is not None:
        gpu_configs = [gpu for gpu in gpu_configs if gpu["vram_gb"] >= min_vram]
    
    if precision:
        gpu_configs = [gpu for gpu in gpu_configs 
                     if precision.lower() in [p.lower() for p in gpu["supported_precisions"]]]
    
    if family:
        gpu_configs = [gpu for gpu in gpu_configs if gpu["family"].lower() == family.lower()]
    
    if generation:
        gpu_configs = [gpu for gpu in gpu_configs if gpu["generation"].lower() == generation.lower()]
    
    # Output results
    if format == 'json':
        click.echo(json.dumps(gpu_configs, indent=2))
    else:
        if len(gpu_configs) == 0:
            click.echo("No GPUs match the specified criteria.")
            return
        
        # Construct filter description
        filters = []
        if min_vram is not None:
            filters.append(f"min VRAM: {min_vram} GB")
        if precision:
            filters.append(f"precision: {precision}")
        if family:
            filters.append(f"family: {family}")
        if generation:
            filters.append(f"generation: {generation}")
        
        filter_desc = ", ".join(filters) if filters else "none"
        click.echo(f"\nGPUs matching filters ({filter_desc}):")
        
        table_data = []
        for gpu in gpu_configs:
            precisions = ", ".join(gpu["supported_precisions"])
            table_data.append([
                gpu["name"],
                gpu["family"],
                gpu["generation"],
                f"{gpu['vram_gb']} GB",
                f"{gpu['fp16_tflops']:.1f}",
                f"{gpu['bandwidth_gb_per_sec']:.1f} GB/s",
                precisions,
                gpu.get("launch_year", "N/A")
            ])
        
        headers = ["Name", "Family", "Generation", "VRAM", "FP16 TFLOPs", "Bandwidth", "Precisions", "Year"]
        click.echo(tabulate(table_data, headers=headers, tablefmt="grid"))

if __name__ == '__main__':
    cli() 