# AdvancedCalculator Class - Usage Guide

This document provides comprehensive documentation for using the `AdvancedCalculator` class directly in your Python code. For API endpoint documentation, refer to the [Calculator API Description](calculator_api_description.md).

## Initializing the Calculator

```python
from src.advanced_calculator import AdvancedCalculator

# Initialize with default settings
calculator = AdvancedCalculator()

# Initialization does not take custom parameters. 
# History tracking is always enabled.
# Precision is specified per calculation method where applicable.
```

## Core Methods

### Model VRAM Calculation

Calculate VRAM requirements for model weights.

```python
model_vram = calculator.calculate_model_vram(
    hidden_dimensions=4096,      # Integer: Hidden dimension size
    feedforward_dimensions=16384, # Integer: Feedforward dimension size
    num_layers=32,               # Integer: Number of transformer layers
    vocab_size=128000,           # Integer: Vocabulary size
    precision="fp16"             # String: Precision format ("fp16", "fp32", "bf16")
)
```

**Returns**: Float - VRAM required for model weights in GB

### KV Cache VRAM Calculation

Calculate VRAM requirements for the KV cache.

```python
kv_cache_vram = calculator.calculate_kv_cache_vram(
    hidden_dimensions=4096,     # Integer: Hidden dimension size
    num_layers=32,              # Integer: Number of transformer layers
    sequence_length=2048,       # Integer: Maximum sequence length
    batch_size=1,               # Integer: Batch size
    precision="fp16"            # String: Precision format ("fp16", "fp32", "bf16")
)
```

**Returns**: Float - VRAM required for KV cache in GB

### Activations VRAM Calculation

Calculate VRAM requirements for activations.

```python
activations_vram = calculator.calculate_activations_vram(
    hidden_dimensions=4096,     # Integer: Hidden dimension size
    feedforward_dimensions=16384, # Integer: Feedforward dimension size
    sequence_length=2048,       # Integer: Maximum sequence length
    batch_size=1,               # Integer: Batch size
    precision="fp16"            # String: Precision format ("fp16", "fp32", "bf16")
)
```

**Returns**: Float - VRAM required for activations in GB

### Total VRAM Calculation

Calculate total VRAM requirements including weights, KV cache, and activations.

```python
total_vram = calculator.calculate_total_vram(
    hidden_dimensions=4096,     # Integer: Hidden dimension size
    feedforward_dimensions=16384, # Integer: Feedforward dimension size
    num_layers=32,              # Integer: Number of transformer layers
    vocab_size=128000,          # Integer: Vocabulary size
    sequence_length=2048,       # Integer: Maximum sequence length
    batch_size=1,               # Integer: Batch size
    precision="fp16",           # String: Precision format ("fp16", "fp32", "bf16")
    weights_overhead=1.05,      # Float: Overhead factor for weights (optional, default: 1.05)
    kv_cache_overhead=1.05,     # Float: Overhead factor for KV cache (optional, default: 1.05)
    activations_overhead=1.1,   # Float: Overhead factor for activations (optional, default: 1.1)
    system_overhead=1.05        # Float: System-wide overhead factor (optional, default: 1.05)
)
```

**Returns**: Dictionary containing the following keys:
```python
{
    "weights_base": float,          # Base VRAM for weights in GB
    "kv_cache_base": float,         # Base VRAM for KV cache in GB
    "activations_base": float,      # Base VRAM for activations in GB
    "weights_with_overhead": float, # VRAM for weights with overhead in GB
    "kv_cache_with_overhead": float, # VRAM for KV cache with overhead in GB
    "activations_with_overhead": float, # VRAM for activations with overhead in GB
    "component_subtotal": float,    # Total VRAM with component overheads in GB
    "total": float                  # Total VRAM with system-wide overhead in GB
}
```

### FLOPs Calculations

Calculate FLOPs for different components of the model.

```python
# Calculate FLOPs for attention mechanism
attention_flops = calculator.calculate_flops_attention(
    batch_size=1,               # Integer: Batch size
    sequence_length=2048,      # Integer: Sequence length
    hidden_dimensions=4096    # Integer: Hidden dimension size
)

# Calculate FLOPs for feedforward network
feedforward_flops = calculator.calculate_flops_feedforward(
    batch_size=1,                # Integer: Batch size
    sequence_length=2048,       # Integer: Sequence length
    hidden_dimensions=4096,     # Integer: Hidden dimension size
    feedforward_dimensions=16384 # Integer: Feedforward dimension size
)

# Calculate FLOPs for prefill phase
prefill_flops = calculator.calculate_flops_prefill(
    batch_size=1,                # Integer: Batch size
    sequence_length=2048,       # Integer: Sequence length
    hidden_dimensions=4096,     # Integer: Hidden dimension size
    feedforward_dimensions=16384, # Integer: Feedforward dimension size
    num_layers=32              # Integer: Number of transformer layers
)

# Calculate FLOPs per token
flops_per_token = calculator.calculate_flops_per_token(
    batch_size=1,               # Integer: Batch size
    hidden_dimensions=4096,     # Integer: Hidden dimension size
    feedforward_dimensions=16384, # Integer: Feedforward dimension size
    num_layers=32              # Integer: Number of transformer layers
)
```

**Returns**: Integer - FLOPs count for the respective calculation

### Throughput and Latency

Calculate throughput and latency metrics.

```python
# Estimate inference throughput (tokens per second)
tokens_per_second = calculator.estimate_inference_throughput(
    flops_per_token=375000000,  # Integer/Float: FLOPs per token
    gpu_tflops=312.0,           # Float: GPU performance in TFLOPS
    efficiency_factor=0.3       # Float: Efficiency factor (0.0-1.0, optional, default: 0.3)
)

# Estimate batch throughput
batch_throughput_info = calculator.estimate_batch_throughput(
    batch_size=4,               # Integer: Batch size
    flops_per_token=375000000,  # Integer: FLOPs per token
    gpu_tflops=312.0,           # Float: GPU performance in TFLOPS
    num_gpus=1,                 # Integer: Number of GPUs (optional, default: 1)
    parallel_efficiency=0.9,    # Float: Efficiency for multi-GPU scaling (optional, default: 0.9)
    compute_efficiency=0.3      # Float: Computation efficiency factor (optional, default: 0.3)
)

# Calculate throughput across various predefined GPUs
throughput_by_gpu = calculator.calculate_throughput_across_gpus(
    flops_per_token=375000000,  # Integer: FLOPs per token
    efficiency_factor=0.3       # Float: Efficiency factor (optional, default: 0.3)
)

# Calculate prefill latency
prefill_latency = calculator.calculate_prefill_latency(
    flops_prefill=1000000000000, # Integer/Float: Total prefill FLOPs
    gpu_tflops=312.0,            # Float: GPU performance in TFLOPS
    efficiency_factor=0.3        # Float: Efficiency factor (0.0-1.0, optional, default: 0.3)
)

# Calculate token generation latency
token_latency = calculator.calculate_token_latency(
    flops_per_token=375000000,  # Integer/Float: FLOPs per token
    gpu_tflops=312.0,           # Float: GPU performance in TFLOPS
    efficiency_factor=0.3       # Float: Efficiency factor (0.0-1.0, optional, default: 0.3)
)

# Calculate total completion latency
completion_latency_info = calculator.calculate_completion_latency(
    prompt_length=512,          # Integer: Number of tokens in the prompt
    output_length=1000,         # Integer: Number of tokens to generate
    flops_prefill=1000000000000, # Integer: Total prefill FLOPs
    flops_per_token=375000000,  # Integer: FLOPs per token
    gpu_tflops=312.0,           # Float: GPU performance in TFLOPS
    efficiency_factor=0.3       # Float: Efficiency factor (optional, default: 0.3)
)
```

**Returns**: 
- `estimate_inference_throughput`: Float - Tokens per second.
- `estimate_batch_throughput`: Dictionary - Throughput information for the batch.
- `calculate_throughput_across_gpus`: Dictionary - Throughput estimates (tokens/sec) keyed by GPU name.
- `calculate_prefill_latency`: Float - Time in seconds.
- `calculate_token_latency`: Float - Time in seconds.
- `calculate_completion_latency`: Dictionary - Latency breakdown (prefill, generation, total).

## High-Level Analysis Methods

### Analyze Model by Name

Analyze a model using a predefined model name.

```python
analysis = calculator.analyze_model_by_name(
    model_name="llama2-7b",     # String: Predefined model name
    sequence_length=2048,       # Integer: Maximum sequence length (optional, uses model default if None)
    batch_size=1,               # Integer: Batch size (optional, default: 1)
    precision="fp16",           # String: Precision format ("fp16", "fp32", "bf16", optional, default: "fp16")
    gpu_tflops=312.0,           # Float: GPU performance in TFLOPS (optional, default: 312.0, A100-like)
    efficiency_factor=0.3       # Float: Efficiency factor (0.0-1.0, optional, default: 0.3)
)
```

**Returns**: Dictionary containing comprehensive analysis results:
```python
{
    "model_info": {  # Details about the model
        "name": str,
        "family": str,
        "parameters_b": float,
        "hidden_dimensions": int,
        "feedforward_dimensions": int,
        "num_layers": int,
        "vocab_size": int,
        "description": str
    },
    "analysis_parameters": { # Parameters used for this analysis
        "sequence_length": int,
        "batch_size": int,
        "precision": str,
        "gpu_tflops": float,
        "efficiency_factor": float
    },
    "flops": {  # Computational requirements
        "attention": int,
        "feedforward": int,
        "prefill_total": int,
        "per_token": int
    },
    "vram": {  # VRAM requirements (matches calculate_total_vram output)
        "weights_base": float,
        "kv_cache_base": float,
        # ... other keys from calculate_total_vram ...
        "total": float 
    },
    "performance": {  # Performance metrics
        "tokens_per_second": float,
        "prefill_latency": float,
        "token_latency": float,
        "time_for_1000_tokens": float,
        "throughput_by_gpu": dict # Throughput on various GPUs
    },
    "overheads_used": {  # Overhead factors applied
        "weights": float,
        "kv_cache": float,
        "activations": float,
        "system": float
    }
}
```

### Analyze Model on GPU

Analyze a model on a specific GPU using predefined configurations.

```python
analysis = calculator.analyze_model_on_gpu(
    model_name="llama2-7b",     # String: Predefined model name
    gpu_name="a100-80gb",       # String: GPU name from predefined configurations
    sequence_length=2048,       # Integer: Maximum sequence length (optional, uses model default if None)
    batch_size=1,               # Integer: Batch size (optional, default: 1)
    precision="fp16",           # String: Precision format ("fp16", "fp32", "bf16", "int8", "int4", optional, default: "fp16")
    efficiency_factor=0.3       # Float: Efficiency factor (0.0-1.0, optional, default: 0.3)
)
```

**Returns**: Dictionary containing comprehensive analysis results, similar to `analyze_model_by_name` but including GPU-specific details:
```python
{
    "model_info": { ... },      # Same as analyze_model_by_name
    "gpu_info": {               # Details about the target GPU
        "name": str,
        "family": str,
        "vram_gb": float,
        "tflops": float,        # TFLOPS for the specified precision
        "supported_precisions": list[str]
    },
    "analysis_parameters": { ... }, # Same as analyze_model_by_name
    "flops": { ... },           # Same as analyze_model_by_name
    "vram": { ... },           # Same as analyze_model_by_name
    "performance": { ... },     # Same as analyze_model_by_name
    "compatibility": {          # Information about fitting the model on the GPU
        "fits_on_gpu": bool,
        "vram_utilization_pct": float,
        "headroom_gb": float,
        "minimum_required_vram_gb": float
    },
    "overheads_used": { ... }   # Same as analyze_model_by_name
}
```

### Determine Model Scaling

Analyze how a model would scale across multiple GPUs.

```python
scaling_result = calculator.determine_model_scaling(
    gpu_vram_gb=80.0,           # Float: VRAM capacity of GPU in GB
    batch_size=1,               # Integer: Batch size
    sequence_length=2048,       # Integer: Maximum sequence length
    hidden_dimensions=4096,     # Integer: Hidden dimension size
    feedforward_dimensions=16384, # Integer: Feedforward dimension size
    num_layers=32,              # Integer: Number of transformer layers
    vocab_size=128000,          # Integer: Vocabulary size
    precision="fp16"            # String: Precision format ("fp16", "fp32", "bf16", optional, default: "fp16")
)
```

**Returns**: Dictionary containing scaling analysis results:
```python
{
    "total_vram_required_gb": float,  # Total VRAM required (calculated)
    "gpu_vram_capacity_gb": float,    # Provided GPU VRAM capacity
    "fits_on_single_gpu": bool,       # Whether model fits on a single GPU
    "num_gpus_required": int,         # Estimated number of GPUs required using Tensor Parallel
    "recommended_strategy": str,      # Recommended parallelism strategy (e.g., "Tensor Parallel", "Pipeline Parallel", "Single GPU")
    "scaling_details": {              # Detailed scaling information
        "vram_per_gpu_tensor_parallel_gb": float, # Estimated VRAM per GPU with Tensor Parallel
        "vram_per_gpu_pipeline_parallel_gb": float, # Estimated VRAM per GPU with Pipeline Parallel (if applicable)
        "model_weights_gb": float,          # VRAM needed for model weights
        "kv_cache_gb": float,               # VRAM needed for KV cache
        "activation_memory_gb": float,      # VRAM needed for activations
        "other_memory_gb": float            # VRAM needed for other components (gradients, optimizer state - often 0 for inference)
    }
}
```

## Working with Predefined Models and GPUs

### Get Model Configurations

Access predefined model configurations.

```python
# Get configuration for a specific model
model_config = calculator.get_model_config("llama2-7b")

# List all available predefined model names
all_model_names = calculator.get_available_models()

# List all available model families
model_families = calculator.get_model_families()

# Get all models from a specific family (returns list of config dicts)
llama_models = calculator.get_models_by_family("Llama")
```

### Get GPU Configurations

Access predefined GPU configurations.

```python
# Get configuration for a specific GPU
gpu_config = calculator.get_gpu_config("a100-80gb")

# List all available predefined GPU names
all_gpu_names = calculator.get_available_gpus()

# List all available GPU families
gpu_families = calculator.get_gpu_families()

# List all available GPU generations
gpu_generations = calculator.get_gpu_generations()

# Get all GPUs from a specific family (returns list of config dicts)
nvidia_gpus = calculator.get_gpus_by_family("NVIDIA")

# Get all GPUs from a specific generation (returns list of config dicts)
ampere_gpus = calculator.get_gpus_by_generation("Ampere")

# Get GPUs with minimum VRAM (returns list of config dicts)
gpus_80gb_plus = calculator.get_gpus_by_min_vram(80.0)

# Get GPUs supporting a specific precision (returns list of config dicts)
bf16_gpus = calculator.get_gpus_supporting_precision("bf16")

# Get recommended GPUs for a model (by name or VRAM requirement)
recommended_gpus = calculator.get_recommended_gpus_for_model(
    model_name_or_vram="llama2-70b", # Can be model name (str) or VRAM in GB (float)
    min_vram_headroom_gb=2.0         # Optional minimum headroom (default: 2.0 GB)
)
```

## Calculation History

The calculator maintains a history of calculations that can be accessed:

```python
# Get full calculation history
history = calculator.get_history()

# Clear calculation history
calculator.clear_history()
```

## Error Handling

The calculator methods will raise `ValueError` exceptions for invalid inputs. It's recommended to use try-except blocks:

```python
try:
    result = calculator.analyze_model_by_name("llama2-7b")
except ValueError as e:
    print(f"Error analyzing model: {e}")
```

## Example Workflow

Here's a complete example workflow:

```python
from src.advanced_calculator import AdvancedCalculator

# Initialize calculator
calc = AdvancedCalculator()

try:
    # Analyze a predefined model on a specific GPU
    analysis = calc.analyze_model_on_gpu(
        model_name="llama2-7b",
        gpu_name="a100-80gb",
        sequence_length=4096,
        batch_size=1,
        precision="fp16"
    )
    
    # Extract key metrics
    total_vram = analysis["vram"]["total_system_wide"]
    tokens_per_second = analysis["performance"]["tokens_per_second"]
    
    print(f"Model requires {total_vram:.2f} GB of VRAM")
    print(f"Expected throughput: {tokens_per_second:.2f} tokens/second")
    
    # Check if model scaling is needed
    if not analysis["compatibility"]["fits_on_gpu"]:
        gpu_vram = analysis["gpu_info"]["vram_gb"]
        model_info = analysis["model_info"]
        scaling = calc.determine_model_scaling(
            gpu_vram_gb=gpu_vram,
            batch_size=analysis["analysis_parameters"]["batch_size"],
            sequence_length=analysis["analysis_parameters"]["sequence_length"],
            hidden_dimensions=model_info["hidden_dimensions"],
            feedforward_dimensions=model_info["feedforward_dimensions"],
            num_layers=model_info["num_layers"],
            vocab_size=model_info["vocab_size"],
            precision=analysis["analysis_parameters"]["precision"]
        )
        
        print(f"Model requires {scaling['num_gpus_required']} GPUs (estimated)")
        print(f"Recommended parallelism strategy: {scaling['recommended_strategy']}")
    
except ValueError as e:
    print(f"Error in calculation: {e}") 