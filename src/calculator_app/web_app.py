#!/usr/bin/env python3
import sys
import os
import logging
import json
import re
import datetime
from flask import Flask, render_template, request, jsonify

# Removed sys.path modification
from .calculator import AdvancedCalculator

# Try to import GPU configurations (simplified)
try:
    # Assuming 'src' is in PYTHONPATH due to how the app is run ('python -m ...')
    # or modules might be moved later.
    # Corrected import path relative to src/
    from src.advanced_calculator.modules.gpus import (
        KNOWN_GPUS,
        get_gpu_config,
        get_gpu_families,
        list_all_gpus
    )
    HAS_GPU_MODULE = True
except ImportError:
    print("Warning: Could not import GPU module from advanced_calculator.modules.gpus. Using fallback.", file=sys.stderr)
    HAS_GPU_MODULE = False

# Try to import model configurations (simplified)
try:
    # Corrected import path relative to src/
    from src.advanced_calculator.modules.models import (
        KNOWN_MODELS,
        get_model_config,
        get_model_families as get_model_families_from_module,
        list_all_models
    )
    HAS_MODELS_MODULE = True
except ImportError:
    print("Warning: Could not import Models module from advanced_calculator.modules.models. Using fallback.", file=sys.stderr)
    HAS_MODELS_MODULE = False

# Initialize Flask, specifying template and static folder locations relative to this file
app = Flask(__name__, template_folder='../../templates', static_folder='../../static')

# Setup logging for the web app
log_dir = "logging"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
web_log_file = os.path.join(log_dir, "web_app_calculations.log")

# Use Flask's built-in logger
file_handler = logging.FileHandler(web_log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

# Initialize calculator using the corrected import
calculator = AdvancedCalculator()

# Add helper methods for performance estimation if they don't exist in the calculator
def estimate_inference_throughput(flops_per_token, gpu_tflops, efficiency_factor=0.3):
    """
    Estimate tokens per second based on per-token FLOPs and GPU performance.
    
    Args:
        flops_per_token: FLOPs required to generate a single token
        gpu_tflops: GPU performance in teraFLOPs
        efficiency_factor: Real-world efficiency factor (0.0-1.0, typically 0.2-0.3)
        
    Returns:
        Estimated tokens per second throughput
    """
    # Convert GPU TFLOPs to FLOPs
    gpu_flops = gpu_tflops * 1e12
    # Apply efficiency factor
    effective_flops = gpu_flops * efficiency_factor
    # Calculate tokens per second
    tokens_per_second = effective_flops / flops_per_token
    return tokens_per_second

def estimate_prefill_latency(flops_prefill, gpu_tflops, efficiency_factor=0.3):
    """
    Estimate time to process the full context (prefill phase).
    
    Args:
        flops_prefill: Total FLOPs for prefill phase (processing full sequence)
        gpu_tflops: GPU performance in teraFLOPs
        efficiency_factor: Real-world efficiency factor (0.0-1.0, typically 0.2-0.3)
        
    Returns:
        Estimated latency in seconds
    """
    # Convert GPU TFLOPs to FLOPs
    gpu_flops = gpu_tflops * 1e12
    # Apply efficiency factor
    effective_flops = gpu_flops * efficiency_factor
    # Calculate latency in seconds
    latency = flops_prefill / effective_flops
    return latency

def estimate_token_generation_latency(flops_per_token, gpu_tflops, efficiency_factor=0.3):
    """
    Estimate time to generate a single token.
    
    Args:
        flops_per_token: FLOPs required to generate a single token
        gpu_tflops: GPU performance in teraFLOPs
        efficiency_factor: Real-world efficiency factor (0.0-1.0, typically 0.2-0.3)
        
    Returns:
        Estimated latency in seconds per token
    """
    # Convert GPU TFLOPs to FLOPs
    gpu_flops = gpu_tflops * 1e12
    # Apply efficiency factor
    effective_flops = gpu_flops * efficiency_factor
    # Calculate latency in seconds
    latency = flops_per_token / effective_flops
    return latency

# Patch the calculator object if these methods don't exist
if not hasattr(calculator, 'estimate_inference_throughput'):
    calculator.estimate_inference_throughput = estimate_inference_throughput

if not hasattr(calculator, 'estimate_prefill_latency'):
    calculator.estimate_prefill_latency = estimate_prefill_latency
    
if not hasattr(calculator, 'estimate_token_generation_latency'):
    calculator.estimate_token_generation_latency = estimate_token_generation_latency

# Fallback GPU configurations if not available from gpus.py
DEFAULT_GPU_CONFIGS = [
    {"name": "A100 (40GB)", "tflops": 312.0, "vram": 40},
    {"name": "A100 (80GB)", "tflops": 312.0, "vram": 80},
    {"name": "H100", "tflops": 756.0, "vram": 80},
    {"name": "H200", "tflops": 989.0, "vram": 141}
]

# Fallback model configurations if not available from calculator
DEFAULT_MODELS = [
    {"name": "Small", "hidden_dim": 768, "ff_dim": 3072, "num_layers": 12, "vocab_size": 50257, "seq_len": 2048, "family": "Default", "parameter_count": 0.125},
    {"name": "Medium", "hidden_dim": 1024, "ff_dim": 4096, "num_layers": 24, "vocab_size": 50257, "seq_len": 4096, "family": "Default", "parameter_count": 0.35},
    {"name": "Large", "hidden_dim": 4096, "ff_dim": 16384, "num_layers": 32, "vocab_size": 120000, "seq_len": 8192, "family": "Default", "parameter_count": 6.7},
    {"name": "XL", "hidden_dim": 8192, "ff_dim": 28672, "num_layers": 80, "vocab_size": 128000, "seq_len": 8192, "family": "Default", "parameter_count": 68.5}
]

# Additional model configurations for popular models
ADDITIONAL_MODELS = [
    {"name": "llama2-7b", "hidden_dim": 4096, "ff_dim": 11008, "num_layers": 32, "vocab_size": 32000, "seq_len": 4096, "family": "Llama", "parameter_count": 7.0},
    {"name": "llama2-13b", "hidden_dim": 5120, "ff_dim": 13824, "num_layers": 40, "vocab_size": 32000, "seq_len": 4096, "family": "Llama", "parameter_count": 13.0},
    {"name": "llama2-70b", "hidden_dim": 8192, "ff_dim": 28672, "num_layers": 80, "vocab_size": 32000, "seq_len": 4096, "family": "Llama", "parameter_count": 70.0},
    {"name": "llama3-8b", "hidden_dim": 4096, "ff_dim": 14336, "num_layers": 32, "vocab_size": 128000, "seq_len": 8192, "family": "Llama", "parameter_count": 8.0},
    {"name": "llama3-70b", "hidden_dim": 8192, "ff_dim": 28672, "num_layers": 80, "vocab_size": 128000, "seq_len": 8192, "family": "Llama", "parameter_count": 70.6},
    {"name": "mistral-7b", "hidden_dim": 4096, "ff_dim": 14336, "num_layers": 32, "vocab_size": 32000, "seq_len": 8192, "family": "Mistral", "parameter_count": 7.3},
    {"name": "phi-2", "hidden_dim": 2560, "ff_dim": 10240, "num_layers": 32, "vocab_size": 51200, "seq_len": 2048, "family": "Phi", "parameter_count": 2.7},
    {"name": "phi-3-mini", "hidden_dim": 3072, "ff_dim": 12288, "num_layers": 32, "vocab_size": 100000, "seq_len": 8192, "family": "Phi", "parameter_count": 3.8}
]

def get_gpu_configs():
    """
    Get GPU configurations from gpus.py module or fallback to defaults.
    
    This function attempts to load GPU configurations from the gpus.py module.
    If that fails, it falls back to default configurations.
    
    Returns:
        List of dictionaries containing GPU configurations with:
        - name: GPU model name
        - tflops: Performance in teraFLOPs
        - vram: VRAM capacity in GB
        - family: GPU family/architecture
        - supported_precisions: List of supported precision formats
    """
    try:
        if HAS_GPU_MODULE:
            # Get all available GPUs from the module
            gpu_list = list_all_gpus()
            result = []
            
            for gpu_id in gpu_list:
                gpu_config = get_gpu_config(gpu_id)
                if gpu_config:
                    # Convert to format expected by the UI
                    result.append({
                        "name": gpu_config["name"],
                        "tflops": gpu_config.get("fp16_tflops", gpu_config.get("bf16_tflops", 0)),
                        "vram": gpu_config.get("vram_gb", 0),
                        "family": gpu_config.get("family", ""),
                        "gen": gpu_config.get("gen", ""),
                        "id": gpu_id,
                        "supported_precisions": gpu_config.get("supported_precisions", ["fp32", "fp16", "bf16"])
                    })
            
            # Sort by family and VRAM
            result.sort(key=lambda x: (x["family"], -x["vram"]))
            return result
        
        # Try to get from calculator if gpus.py module is not available
        if hasattr(calculator, 'get_gpu_configs'):
            return calculator.get_gpu_configs()
            
        # Add supported_precisions to default configs
        return [
            {"name": "A100 (40GB)", "tflops": 312.0, "vram": 40, "supported_precisions": ["fp32", "tf32", "fp16", "bf16", "int8", "int4"]},
            {"name": "A100 (80GB)", "tflops": 312.0, "vram": 80, "supported_precisions": ["fp32", "tf32", "fp16", "bf16", "int8", "int4"]},
            {"name": "H100", "tflops": 756.0, "vram": 80, "supported_precisions": ["fp32", "tf32", "fp16", "bf16", "fp8", "int8", "int4"]},
            {"name": "H200", "tflops": 989.0, "vram": 141, "supported_precisions": ["fp32", "tf32", "fp16", "bf16", "fp8", "int8", "int4"]}
        ]
    except Exception as e:
        print(f"Error getting GPU configs: {e}")
        # Add supported_precisions to default configs
        return [
            {"name": "A100 (40GB)", "tflops": 312.0, "vram": 40, "supported_precisions": ["fp32", "tf32", "fp16", "bf16", "int8", "int4"]},
            {"name": "A100 (80GB)", "tflops": 312.0, "vram": 80, "supported_precisions": ["fp32", "tf32", "fp16", "bf16", "int8", "int4"]},
            {"name": "H100", "tflops": 756.0, "vram": 80, "supported_precisions": ["fp32", "tf32", "fp16", "bf16", "fp8", "int8", "int4"]},
            {"name": "H200", "tflops": 989.0, "vram": 141, "supported_precisions": ["fp32", "tf32", "fp16", "bf16", "fp8", "int8", "int4"]}
        ]

def get_available_models():
    """
    Get available models from models.py module or fallback to defaults.
    
    This function attempts to load model configurations from the models.py module.
    If that fails, it falls back to default and additional model configurations.
    
    Returns:
        List of dictionaries containing model configurations with:
        - name: Model name
        - hidden_dim: Hidden layer dimensions
        - ff_dim: Feedforward layer dimensions
        - num_layers: Number of transformer layers
        - vocab_size: Vocabulary size
        - seq_len: Default sequence length
        - family: Model family/architecture
        - parameter_count: Billions of parameters
    """
    try:
        # First try to get models from models.py
        if HAS_MODELS_MODULE:
            model_list = list_all_models()
            result = []
            
            for model_id in model_list:
                model_config = get_model_config(model_id)
                if model_config:
                    # Convert to format expected by the UI
                    result.append({
                        "name": model_config["name"],
                        "hidden_dim": model_config.get("hidden_dimensions"),
                        "ff_dim": model_config.get("feedforward_dimensions"),
                        "num_layers": model_config.get("num_layers"),
                        "vocab_size": model_config.get("vocab_size", 50257),
                        "seq_len": model_config.get("default_seq_length", 2048),
                        "family": model_config.get("family", ""),
                        "parameter_count": model_config.get("parameter_count", 0)
                    })
            
            if result and len(result) > 5:  # If we have a reasonable number of models
                return result
        
        # Then try to get models from calculator
        if hasattr(calculator, 'get_available_models'):
            models = calculator.get_available_models()
            if models and len(models) > 5:  # If we have a reasonable number of models
                return models
            
        # If not available or too few models, use the default and additional models
        return DEFAULT_MODELS + ADDITIONAL_MODELS
    except Exception as e:
        print(f"Error getting models: {e}")
        return DEFAULT_MODELS + ADDITIONAL_MODELS

def get_model_families():
    """
    Get model families from models.py module or fallback to defaults.
    
    This function attempts to get the list of model families for organizing
    the dropdown menu in the UI.
    
    Returns:
        List of model family names (strings)
    """
    try:
        # First try to get families from models.py
        if HAS_MODELS_MODULE:
            families = get_model_families_from_module()
            if families and len(families) > 1:  # If we have real families
                return families
        
        # Then try to get families from calculator
        if hasattr(calculator, 'get_model_families'):
            families = calculator.get_model_families()
            if families and len(families) > 1:  # If we have real families
                return families
        
        # Otherwise extract unique families from our models
        models = get_available_models()
        families = set()
        for model in models:
            if model.get('family'):
                families.add(model.get('family'))
        
        return sorted(list(families))
    except Exception as e:
        print(f"Error getting model families: {e}")
        return ["Default", "Llama", "Mistral", "Phi"]

@app.route('/')
def index():
    """
    Render the main calculator interface.
    
    This route loads all necessary data for the calculator interface,
    including GPU configurations, model families, and available models.
    
    Returns:
        Rendered HTML template with data for the calculator UI
    """
    gpu_configs = get_gpu_configs()
    model_families = get_model_families()
    models = get_available_models()
    
    return render_template('index.html', 
                          gpu_configs=gpu_configs,
                          model_families=model_families,
                          models=models)

@app.route('/api/calculate', methods=['POST'])
def calculate():
    """
    API endpoint to perform model calculations.
    
    This route accepts model and hardware configuration parameters via POST request
    and returns a complete analysis of the model's computational requirements, 
    memory usage, and performance estimates.
    
    Expected POST JSON payload:
    {
        "model_name": "model_name_or_custom",
        "hidden_dim": int,
        "ff_dim": int,
        "num_layers": int,
        "vocab_size": int,
        "seq_len": int,
        "batch_size": int,
        "gpu_tflops": float,
        "gpu_id": string (optional),
        "efficiency": float (0.0-1.0),
        "precision": "fp16"/"bf16"/"fp32",
        "parallelism_strategy": string ("none"/"tp"/"pp"),
        "tp_size": int,
        "pp_size": int,
        "num_gpus": int
    }
    
    Returns:
        JSON with:
        - model_name: Selected model name
        - parameters_billions: Total parameters in billions
        - flops: Computational requirements breakdown
        - vram: Memory requirements breakdown
        - performance: Throughput and latency estimates
        - parallelism: Information about the applied parallelism
    """
    data = request.json
    app.logger.info(f"Received calculation request: {json.dumps(data)}")

    try:
        # Extract parameters from request
        model_name = data.get('model_name', 'Custom')
        hidden_dim = int(data.get('hidden_dim', 768))
        ff_dim = int(data.get('ff_dim', 3072))
        num_layers = int(data.get('num_layers', 12))
        vocab_size = int(data.get('vocab_size', 50257))
        seq_len = int(data.get('seq_len', 2048))
        batch_size = int(data.get('batch_size', 1))
        gpu_tflops = float(data.get('gpu_tflops', 312.0))
        gpu_id = data.get('gpu_id', None)
        efficiency = float(data.get('efficiency', 0.3))
        precision = data.get('precision', 'fp16')
        # Extract parallelism settings
        parallelism_strategy = data.get('parallelism_strategy', 'none')
        tp_size = int(data.get('tp_size', 1))
        pp_size = int(data.get('pp_size', 1))
        num_gpus = int(data.get('num_gpus', 1))
        
        # If GPU ID is provided and gpus module is available, use it to get more accurate GPU info
        if gpu_id and HAS_GPU_MODULE:
            try:
                gpu_config = get_gpu_config(gpu_id)
                if gpu_config:
                    # Update TFLOPs based on precision
                    if precision == 'fp16' and 'fp16_tflops' in gpu_config:
                        gpu_tflops = gpu_config['fp16_tflops']
                    elif precision == 'bf16' and 'bf16_tflops' in gpu_config:
                        gpu_tflops = gpu_config['bf16_tflops']
                    elif precision == 'fp32' and 'fp32_tflops' in gpu_config:
                        gpu_tflops = gpu_config['fp32_tflops']
            except Exception as e:
                print(f"Error getting GPU config for {gpu_id}: {e}")
        
        # Calculate FLOPs
        flops_attention = calculator.calculate_flops_attention(batch_size, seq_len, hidden_dim)
        flops_feedforward = calculator.calculate_flops_feedforward(batch_size, seq_len, hidden_dim, ff_dim)
        flops_prefill = calculator.calculate_flops_prefill(batch_size, seq_len, hidden_dim, ff_dim, num_layers)
        
        # Calculate per-token FLOPs (for a single new token)
        flops_per_token = (
            calculator.calculate_flops_attention(batch_size, 1, hidden_dim) +
            calculator.calculate_flops_feedforward(batch_size, 1, hidden_dim, ff_dim)
        ) * num_layers
        
        # VRAM calculations
        model_vram = calculator.calculate_model_vram(
            hidden_dim, ff_dim, num_layers, vocab_size, precision
        )
        
        kv_cache_vram = calculator.calculate_kv_cache_vram(
            batch_size, seq_len, hidden_dim, num_layers, precision
        )
        
        # Calculate activations VRAM if the method exists
        try:
            activations_vram = calculator.calculate_activations_vram(
                batch_size, seq_len, hidden_dim, num_layers, precision
            )
        except:
            # Estimate activations as 10% of model size for fallback
            activations_vram = model_vram * 0.1
        
        # Calculate total VRAM with breakdown
        weights_overhead = 1.05  # 5% overhead for model weights
        kv_cache_overhead = 1.05  # 5% overhead for KV cache
        activations_overhead = 1.1  # 10% overhead for activations
        system_overhead = 1.05  # 5% overhead for system
        
        # Apply overheads
        model_vram_with_overhead = model_vram * weights_overhead
        kv_cache_with_overhead = kv_cache_vram * kv_cache_overhead
        activations_with_overhead = activations_vram * activations_overhead
        
        # Calculate subtotal before system overhead
        subtotal_vram = model_vram_with_overhead + kv_cache_with_overhead + activations_with_overhead
        
        # Apply system overhead to get total
        total_vram = subtotal_vram * system_overhead
        
        # Try to use the calculator's total_vram method if it exists and produces reasonable results
        try:
            calc_total_vram = calculator.calculate_total_vram(
                batch_size, seq_len, hidden_dim, ff_dim, num_layers, vocab_size, precision,
                weights_overhead=weights_overhead,
                kv_cache_overhead=kv_cache_overhead,
                activations_overhead=activations_overhead,
                system_overhead=system_overhead
            )
            # Use the calculator's total if it seems reasonable
            if calc_total_vram > 0 and calc_total_vram >= model_vram:
                total_vram = calc_total_vram
        except:
            # Keep using our calculated total if there's an error
            pass
        
        # Adjust calculations based on parallelism
        # Note: We might need to refine these adjustments based on specific parallelism implementation details.
        # This is a simplified model.
        
        # Tensor Parallelism (TP): Scales FLOPs/activations/weights across TP GPUs
        # Pipeline Parallelism (PP): Scales stages across PP GPUs, affecting latency
        
        effective_gpu_tflops = gpu_tflops * num_gpus # Total compute power
        flops_per_gpu = gpu_tflops # Compute power per individual GPU

        # VRAM calculations (per GPU)
        # Model weights are sharded with TP
        model_vram_per_gpu = calculator.calculate_model_vram(
            hidden_dim, ff_dim, num_layers, vocab_size, precision
        ) / tp_size 
        
        # KV cache is typically per sequence, not sharded by TP, but check batch size distribution?
        # For simplicity, assume KV cache is replicated or managed per pipeline stage
        kv_cache_vram_per_gpu = calculator.calculate_kv_cache_vram(
            max(1, batch_size // pp_size), seq_len, hidden_dim, num_layers, precision # Ensure min batch size 1
        ) 
        
        # Activations are complex. With TP, they are sharded. With PP, only one stage\'s activations are needed.
        # Simplified: Divide by TP size. PP might further reduce this, but depends on implementation.
        try:
            activations_vram_per_gpu = calculator.calculate_activations_vram(
                max(1, batch_size // pp_size), seq_len, hidden_dim, num_layers, precision # Ensure min batch size 1
            ) / tp_size 
        except:
            activations_vram_per_gpu = (model_vram / tp_size) * 0.1 # Fallback estimate

        # Apply overheads per GPU
        model_vram_per_gpu_with_overhead = model_vram_per_gpu * weights_overhead
        kv_cache_per_gpu_with_overhead = kv_cache_vram_per_gpu * kv_cache_overhead
        activations_per_gpu_with_overhead = activations_vram_per_gpu * activations_overhead
        
        subtotal_vram_per_gpu = model_vram_per_gpu_with_overhead + kv_cache_per_gpu_with_overhead + activations_per_gpu_with_overhead
        total_vram_per_gpu = subtotal_vram_per_gpu * system_overhead
        
        # Performance Estimates
        # Throughput: Scales with total effective FLOPs
        tokens_per_second = calculator.estimate_inference_throughput(
            flops_per_token, effective_gpu_tflops, efficiency 
        )
        
        # Latency: More complex
        # Prefill Latency: Limited by the slowest stage in PP, compute distributed by TP
        prefill_latency_per_stage = calculator.estimate_prefill_latency(
            flops_prefill / pp_size, # FLOPs per PP stage
            flops_per_gpu * tp_size, # FLOPs available per stage (TP combined)
            efficiency
        )
        # Add pipeline bubble latency (simplistic: (PP-1) * time_per_stage)
        # This assumes perfect load balancing and ignores communication overhead
        prefill_latency = prefill_latency_per_stage * pp_size # Simplistic - total time across stages
        # A more accurate model considers the pipeline bubble: prefill_latency_per_stage + (pp_size - 1) * per_token_latency_per_stage
        
        # Token Latency (time per token): Limited by slowest stage, compute distributed by TP
        token_latency_per_stage = calculator.estimate_token_generation_latency(
            flops_per_token / pp_size, # FLOPs per token per PP stage
            flops_per_gpu * tp_size, # FLOPs available per stage (TP combined)
            efficiency
        )
        # In steady state, pipeline produces a token every stage latency
        token_latency = token_latency_per_stage 
        
        # --- End of Parallelism Adjustments ---

        # Get the total parameters
        total_params_billions = None
        
        # First try to get from models.py module if available
        if HAS_MODELS_MODULE and model_name != 'Custom':
            try:
                # Find model by name in all known models
                model_list = list_all_models()
                for model_id in model_list:
                    model_config = get_model_config(model_id)
                    if (model_config and 
                        (model_config.get('name', '').lower() == model_name.lower() or 
                         model_id.lower() == model_name.lower())):
                        total_params_billions = model_config.get('parameter_count')
                        break
            except Exception as e:
                print(f"Error getting parameter count for {model_name}: {e}")
        
        # If not found in models.py, calculate from dimensions
        if total_params_billions is None:
            attention_params = 4 * hidden_dim * hidden_dim * num_layers
            ff_params = 2 * hidden_dim * ff_dim * num_layers
            embedding_params = vocab_size * hidden_dim
            layernorm_params = 4 * hidden_dim * num_layers + 2 * hidden_dim
            total_params = attention_params + ff_params + embedding_params + layernorm_params
            total_params_billions = total_params / 1e9
        
        results = {
            "model_name": model_name,
            "parameters_billions": round(total_params_billions, 2) if total_params_billions else 0,
            "flops": {
                "attention": flops_attention,
                "feedforward": flops_feedforward,
                "prefill": flops_prefill,
                "per_token": flops_per_token
            },
            "vram": {
                "model": model_vram, # Total model VRAM
                "kv_cache": kv_cache_vram, # Total KV cache VRAM
                "activations": activations_vram, # Total Activations VRAM
                "model_per_gpu": model_vram_per_gpu,
                "kv_cache_per_gpu": kv_cache_vram_per_gpu,
                "activations_per_gpu": activations_vram_per_gpu,
                "total_per_gpu": total_vram_per_gpu,
                "model_with_overhead": model_vram_with_overhead, # Total with overhead
                "kv_cache_with_overhead": kv_cache_with_overhead, # Total with overhead
                "activations_with_overhead": activations_with_overhead, # Total with overhead
                "total_system_wide": total_vram # Original total estimate (might need review)
            },
            "performance": {
                "tokens_per_second": tokens_per_second,
                "prefill_latency": prefill_latency,
                "token_latency": token_latency
            },
            "parallelism": {
                "strategy": parallelism_strategy,
                "tp_size": tp_size,
                "pp_size": pp_size,
                "num_gpus": num_gpus,
                "effective_tflops": effective_gpu_tflops
            }
        }

        app.logger.info(f"Calculation successful. Results: {json.dumps(results)}")
        return jsonify(results)

    except (ValueError, TypeError) as e:
        app.logger.error(f"Error during calculation: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.error(f"Unexpected error during calculation: {str(e)}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred"}), 500

@app.route('/api/available_models', methods=['GET'])
def available_models():
    """API endpoint to get available model configurations"""
    try:
        models = get_available_models()
        return jsonify({"models": models})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/model_scaling_analysis', methods=['POST'])
def model_scaling_analysis():
    """
    API endpoint to analyze if a model fits on the selected hardware
    and determine scaling/parallelization strategies if needed.
    
    Request JSON should include:
    - model_config: Model configuration object
    - gpu_config: GPU configuration object
    - batch_size: Batch size for inference
    - sequence_length: Maximum sequence length
    - precision: Model precision format (fp16, fp32, bf16)
    
    Returns JSON with:
    - fits_on_single_gpu: Boolean indicating if model fits on a single GPU
    - num_gpus_required: Number of GPUs needed if parallelization is required
    - recommended_strategy: Parallelization strategy recommendation
    - scaling_details: Additional details about the parallelization
    """
    try:
        # Get request data
        req_data = request.get_json()
        
        # Extract parameters from request
        model_config = req_data.get('model_config', {})
        gpu_config = req_data.get('gpu_config', {})
        batch_size = int(req_data.get('batch_size', 1))
        sequence_length = int(req_data.get('sequence_length', model_config.get('seq_len', 2048)))
        precision = req_data.get('precision', 'fp16')
        
        # Validate required parameters
        if not model_config:
            return jsonify({"error": "Model configuration is required"}), 400
        if not gpu_config:
            return jsonify({"error": "GPU configuration is required"}), 400
            
        # Extract model parameters
        hidden_dim = int(model_config.get('hidden_dim', 0))
        ff_dim = int(model_config.get('ff_dim', 0))
        num_layers = int(model_config.get('num_layers', 0))
        vocab_size = int(model_config.get('vocab_size', 0))
        
        # Calculate total VRAM required
        vram_calc = calculator
        
        total_vram_required = vram_calc.calculate_total_vram(
            batch_size=batch_size,
            sequence_length=sequence_length,
            hidden_dimensions=hidden_dim,
            feedforward_dimensions=ff_dim,
            num_layers=num_layers,
            vocab_size=vocab_size,
            precision=precision
        )
        
        # Get GPU VRAM capacity
        gpu_vram_capacity = float(gpu_config.get('vram', 0))
        
        # Check if calculator has the new method
        if hasattr(vram_calc, 'determine_model_scaling'):
            # Use the new method to determine scaling strategy
            scaling_result = vram_calc.determine_model_scaling(
                gpu_vram_gb=gpu_vram_capacity,
                batch_size=batch_size,
                sequence_length=sequence_length,
                hidden_dimensions=hidden_dim,
                feedforward_dimensions=ff_dim,
                num_layers=num_layers,
                vocab_size=vocab_size,
                precision=precision
            )
            
            # Prepare response
            response = {
                "fits_on_single_gpu": scaling_result["fits_on_single_gpu"],
                "total_vram_required_gb": scaling_result["total_vram_required_gb"],
                "gpu_vram_capacity_gb": gpu_vram_capacity,
                "num_gpus_required": scaling_result["num_gpus_required"],
                "recommended_strategy": scaling_result["recommended_strategy"],
                "scaling_details": {
                    "tensor_parallelism_degree": scaling_result["tp_degree"],
                    "pipeline_parallelism_degree": scaling_result["pp_degree"],
                    "estimated_efficiency": scaling_result["estimated_efficiency"],
                    "vram_per_gpu_gb": scaling_result["vram_per_gpu"],
                    "communication_overhead_gb": scaling_result["communication_overhead_gb"]
                }
            }
        else:
            # Fallback if the method doesn't exist
            fits_on_gpu = total_vram_required <= gpu_vram_capacity
            num_gpus_needed = max(1, int(total_vram_required / gpu_vram_capacity) + 1)
            
            response = {
                "fits_on_single_gpu": fits_on_gpu,
                "total_vram_required_gb": total_vram_required,
                "gpu_vram_capacity_gb": gpu_vram_capacity,
                "num_gpus_required": 1 if fits_on_gpu else num_gpus_needed,
                "recommended_strategy": "single" if fits_on_gpu else "unknown",
                "scaling_details": {
                    "note": "Advanced scaling analysis not available. Please upgrade calculator."
                }
            }
            
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/gpu_configs', methods=['GET'])
def gpu_configs():
    """API endpoint to get all available GPU configurations"""
    try:
        configs = get_gpu_configs()
        return jsonify({"gpu_configs": configs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/model_config/<model_name>', methods=['GET'])
def model_config(model_name):
    """
    API endpoint to get a specific model configuration.
    
    Args:
        model_name: Name of the model to retrieve configuration for
        
    Returns:
        JSON with model configuration details or error message
    """
    try:
        # First try to get from models.py module if available
        if HAS_MODELS_MODULE:
            # Find model by name in all known models
            model_list = list_all_models()
            for model_id in model_list:
                model_config = get_model_config(model_id)
                if (model_config and 
                    (model_config.get('name', '').lower() == model_name.lower() or 
                     model_id.lower() == model_name.lower())):
                    return jsonify(model_config)
        
        # Try to get model config from the calculator if the method exists
        if hasattr(calculator, 'get_model_config'):
            config = calculator.get_model_config(model_name)
            if config:
                return jsonify(config)
        
        # Otherwise search in the default models
        models = get_available_models()
        for model in models:
            if model.get('name').lower() == model_name.lower():
                return jsonify(model)
                
        return jsonify({"error": "Model not found"}), 404
    except Exception as e:
        print(f"Error getting model config for {model_name}: {e}")
        return jsonify({"error": f"Error retrieving model: {str(e)}"}), 404

@app.route('/visualize')
def visualize_results():
    """Render the visualization page with data from logs."""
    cli_log_path = os.path.join(log_dir, "cli_calculations.log")
    web_log_path = os.path.join(log_dir, "web_app_calculations.log")
    
    cli_data = parse_log_file(cli_log_path, 'cli')
    web_data = parse_log_file(web_log_path, 'web')
    
    all_data = cli_data + web_data
    # Re-sort combined list by timestamp descending
    all_data.sort(key=lambda x: datetime.datetime.strptime(x["timestamp"], '%Y-%m-%d %H:%M:%S,%f') if ',' in x["timestamp"] else datetime.datetime.strptime(x["timestamp"], '%Y-%m-%d %H:%M:%S'), reverse=True)
    
    return render_template('visualize.html', logged_data=all_data)

# --- Helper functions for log parsing ---

def extract_data_from_log_line(line, log_type):
    """Extracts input arguments and results JSON from a log line."""
    try:
        if log_type == 'cli':
            # CLI Log Format: DATETIME - LEVEL - Message
            # We look for lines containing Arguments: {...} or Calculation results (JSON): {...}
            match_args = re.search(r"Arguments: ({.*?})$", line)
            match_results = re.search(r"Calculation results \(JSON\): ({.*?})$", line)
            timestamp_str = line.split(" - ")[0]
            
            if match_args:
                data = json.loads(match_args.group(1).replace("'", "\"")) # Handle potential single quotes
                return {"type": "input", "timestamp": timestamp_str, "data": data}
            elif match_results:
                data = json.loads(match_results.group(1))
                return {"type": "output", "timestamp": timestamp_str, "data": data}
                
        elif log_type == 'web':
            # Web Log Format: DATETIME LEVEL: Message [in path:line]
            # Corrected Regex: Look for JSON ending before " [in ..." part
            match_request = re.search(r"Received calculation request: ({.*?})\s+\[in", line)
            match_results = re.search(r"Calculation successful\. Results: ({.*?})\s+\[in", line)
            timestamp_str = line.split(" INFO:")[0].split(" ERROR:")[0].split(" WARNING:")[0] # Extract timestamp

            if match_request:
                data = json.loads(match_request.group(1))
                return {"type": "input", "timestamp": timestamp_str, "data": data}
            elif match_results:
                data = json.loads(match_results.group(1))
                return {"type": "output", "timestamp": timestamp_str, "data": data}
                
    except (json.JSONDecodeError, AttributeError, IndexError) as e:
        # Log parsing errors or ignore malformed lines
        # print(f"Skipping log line due to parsing error: {e} - Line: {line.strip()}")
        pass
    return None

def parse_log_file(log_path, log_type):
    """Reads a log file and parses relevant calculation entries."""
    if not os.path.exists(log_path):
        return []

    parsed_entries = []
    last_input = None
    
    try:
        with open(log_path, 'r') as f:
            for line in f:
                extracted = extract_data_from_log_line(line, log_type)
                if extracted:
                    if extracted["type"] == "input":
                        last_input = extracted
                    elif extracted["type"] == "output" and last_input and last_input["timestamp"] <= extracted["timestamp"]:
                        # Combine last input with this output
                        # A simple heuristic: pair output with the most recent preceding input
                        parsed_entries.append({
                            "timestamp": extracted["timestamp"],
                            "source": log_type.upper(),
                            "inputs": last_input["data"],
                            "outputs": extracted["data"]
                        })
                        last_input = None # Reset last input after pairing
                    elif extracted["type"] == "output":
                         # Output without a recent input found, maybe log it separately?
                         # For now, we only add paired entries
                         pass
    except Exception as e:
        app.logger.error(f"Error reading or parsing log file {log_path}: {e}", exc_info=True)

    # Sort by timestamp descending (most recent first)
    parsed_entries.sort(key=lambda x: datetime.datetime.strptime(x["timestamp"], '%Y-%m-%d %H:%M:%S,%f') if ',' in x["timestamp"] else datetime.datetime.strptime(x["timestamp"], '%Y-%m-%d %H:%M:%S'), reverse=True)
    return parsed_entries

# --- End Helper functions ---

if __name__ == '__main__':
    # Logging is already configured via app.logger
    app.run(debug=True) # debug=True automatically enables Flask's reloader and debugger 