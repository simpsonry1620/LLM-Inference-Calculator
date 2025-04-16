# LLM Inference Calculator API Documentation

## Overview

The LLM Inference Calculator API provides tools for estimating computational requirements, memory usage, and performance characteristics of large language models (LLMs). This API helps determine infrastructure needs for running inference with different model architectures on various GPU hardware.

## Base URL

All API endpoints are relative to the base deployment URL.

## Authentication

Currently, the API does not require authentication.

## API Endpoints

### 1. Calculate Model Requirements

Calculates resource requirements and performance metrics for a specified model configuration.

**Endpoint:** `/api/calculate`  
**Method:** `POST`  
**Content-Type:** `application/json`

#### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `calculation_type` | string | No | `"model"` | Type of calculation to perform (currently only "model" is supported) |
| `model_name` | string | No | `"custom"` | Name of predefined model or "custom" for custom configuration |
| `hidden_dim` | integer | Yes* | - | Hidden dimension size of the model |
| `ff_dim` | integer | Yes* | - | Feedforward dimension size |
| `num_layers` | integer | Yes* | - | Number of transformer layers |
| `vocab_size` | integer | Yes* | - | Vocabulary size of the model |
| `seq_length` | integer | No | 2048 | Maximum sequence length |
| `batch_size` | integer | No | 1 | Batch size for inference |
| `gpu` | string | No | - | ID of GPU to use for calculation (if not provided, uses TFLOPS value) |
| `gpu_tflops` | float | No | 312.0 | GPU performance in TFLOPS (used if no `gpu` provided) |
| `precision` | string | No | `"fp16"` | Model precision (fp16, fp32, bf16) |
| `efficiency_factor` | float | No | 0.3 | GPU efficiency factor (0.0-1.0) |

\* Required for custom models, not needed when using a predefined model name

#### Response Format

```json
{
  "model_name": "custom",
  "parameters_billions": 7.0,
  "vram": {
    "model_base": 14.0,
    "kv_cache_base": 2.0,
    "activations_base": 1.0,
    "model_with_overhead": 15.4,
    "kv_cache_with_overhead": 2.4,
    "activations_with_overhead": 1.2,
    "total_base": 17.0,
    "total_with_component_overhead": 19.0,
    "total_system_wide": 19.0,
    "model_per_gpu": 14.0,
    "kv_cache_per_gpu": 2.0,
    "activations_per_gpu": 1.0,
    "model_per_gpu_with_overhead": 15.4,
    "kv_cache_per_gpu_with_overhead": 2.4,
    "activations_per_gpu_with_overhead": 1.2,
    "total_base_per_gpu": 17.0,
    "total_per_gpu_with_component_overhead": 19.0
  },
  "flops": {
    "attention": 1.23e+12,
    "feedforward": 3.45e+12,
    "prefill_total": 9.87e+12,
    "per_token": 4.56e+09
  },
  "performance": {
    "tokens_per_second": 328.4,
    "prefill_latency": 0.1056,
    "token_latency": 0.0032,
    "time_for_1000_tokens": 3.2,
    "throughput_by_gpu": {
      "A100": 328.4,
      "H100": 756.2
    },
    "bandwidth_utilization": 0.45
  },
  "parallelism": {
    "recommended_strategy": "tensor",
    "tensor_parallel_gpus": 1,
    "pipeline_parallel_gpus": 1
  },
  "overheads_used": {
    "weights": 1.1,
    "kv_cache": 1.2,
    "activations": 1.2
  },
  "history": [
    {"operation": "calculate_model_size", "result": "14 GB", "timestamp": "2023-01-01T12:00:00"},
    {"operation": "calculate_kv_cache", "result": "2 GB", "timestamp": "2023-01-01T12:00:01"}
  ]
}
```

### 2. Get Available Models

Returns a list of predefined models available for analysis.

**Endpoint:** `/api/available_models`  
**Method:** `GET`

#### Response Format

```json
[
  {
    "name": "llama2-7b",
    "hidden_dim": 4096,
    "ff_dim": 11008,
    "num_layers": 32,
    "vocab_size": 32000,
    "seq_len": 4096,
    "family": "Llama",
    "parameter_count": 7.0
  },
  {
    "name": "llama2-13b",
    "hidden_dim": 5120,
    "ff_dim": 13824,
    "num_layers": 40,
    "vocab_size": 32000,
    "seq_len": 4096,
    "family": "Llama",
    "parameter_count": 13.0
  }
]
```

### 3. Model Scaling Analysis

Analyzes how a model would scale across different configurations and GPU setups.

**Endpoint:** `/api/model_scaling_analysis`  
**Method:** `POST`  
**Content-Type:** `application/json`

#### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `gpu_vram_gb` | float | Yes | - | VRAM capacity of GPU in GB |
| `batch_size` | integer | No | 1 | Batch size for inference |
| `sequence_length` | integer | No | 2048 | Maximum sequence length |
| `hidden_dimensions` | integer | Yes | - | Hidden dimension size of the model |
| `feedforward_dimensions` | integer | Yes | - | Feedforward dimension size |
| `num_layers` | integer | Yes | - | Number of transformer layers |
| `vocab_size` | integer | Yes | - | Vocabulary size of the model |
| `precision` | string | No | `"fp16"` | Model precision (fp16, fp32, bf16) |

#### Response Format

```json
{
  "total_vram_required_gb": 24.5,
  "gpu_vram_capacity_gb": 16.0,
  "fits_on_single_gpu": false,
  "num_gpus_required": 2,
  "recommended_strategy": "tensor_parallel",
  "scaling_details": {
    "single_gpu_memory_breakdown": {
      "model_weights_gb": 14.0,
      "kv_cache_gb": 2.0,
      "activation_memory_gb": 1.0,
      "optimizer_states_gb": 0.0,
      "gradient_accumulation_gb": 0.0
    },
    "parallel_strategies": {
      "tensor_parallel": {
        "num_gpus": 2,
        "vram_per_gpu_gb": 12.25,
        "communication_overhead": "low",
        "implementation_complexity": "medium"
      },
      "pipeline_parallel": {
        "num_gpus": 2,
        "vram_per_gpu_gb": 12.25,
        "communication_overhead": "medium",
        "implementation_complexity": "high"
      }
    }
  }
}
```

## Data Types

### Model Parameters

* `hidden_dim`: Integer - Size of the hidden/embedding dimension
* `ff_dim`: Integer - Size of the feedforward/MLP dimension
* `num_layers`: Integer - Number of transformer layers
* `vocab_size`: Integer - Size of the vocabulary
* `seq_length`: Integer - Maximum sequence length supported
* `batch_size`: Integer - Number of sequences processed simultaneously

### GPU Parameters

* `gpu`: String - ID of GPU to analyze (matches the IDs from the GPU configurations)
* `gpu_tflops`: Float - Performance in teraFLOPs per second
* `precision`: String - Precision format for calculation:
  * `"fp32"` - 32-bit floating point (4 bytes per parameter)
  * `"fp16"` - 16-bit floating point (2 bytes per parameter)
  * `"bf16"` - Brain floating point (2 bytes per parameter)
  * `"int8"` - 8-bit integer (1 byte per parameter, only for specific GPUs)
  * `"int4"` - 4-bit integer (0.5 bytes per parameter, only for specific GPUs)
* `efficiency_factor`: Float - Efficiency factor (0.0-1.0) representing how effectively the GPU hardware is utilized

## Error Responses

All errors return a JSON object with an `error` field describing the issue:

```json
{
  "error": "Error message describing the issue"
}
```

Common error scenarios:
- Invalid input parameters (400 Bad Request)
- Internal calculation error (500 Internal Server Error)
- Unsupported calculation type (400 Bad Request)

## Assumptions and Limitations

The calculator makes several assumptions to simplify calculations:

1. Memory requirements:
   - Model weights, KV cache, and activations are the main VRAM components
   - Overheads are applied to account for implementation details
   
2. Performance calculations:
   - Efficiency factor approximates real-world hardware utilization
   - Calculations assume standard decoder-only transformer architecture
   - Results are estimates and may vary from actual implementation performance

3. Precision:
   - Different precision formats affect both memory requirements and computational performance
   - The calculator validates that the selected GPU supports the requested precision

4. Multi-GPU scaling:
   - Scaling calculations account for communication overhead 
   - Different parallelism strategies (tensor, pipeline, sequence) are compared

For a full list of assumptions, refer to the assumptions documentation.

## Calculator Response Formats

The Advanced Calculator provides two main analysis methods that return different response structures. These differences can cause issues when processing the responses in the web UI.

### 1. `analyze_model_by_name` Response Format

```json
{
  "model_info": {
    "name": "model_name",
    "family": "Model Family",
    "parameters_b": 7.0,
    "hidden_dimensions": 4096,
    "feedforward_dimensions": 11008,
    "num_layers": 32,
    "vocab_size": 32000,
    "description": "Model description"
  },
  "analysis_parameters": {
    "sequence_length": 2048,
    "batch_size": 1,
    "precision": "fp16",
    "gpu_tflops": 312.0,
    "efficiency_factor": 0.3
  },
  "flops": {
    "attention": 171798691840,
    "feedforward": 962072674304,
    "prefill_total": 90709709291520,
    "per_token": 42950328320
  },
  "vram": {
    "model_weights": 14.51,
    "kv_cache": 1.0,
    "activations": 2.0,
    "total": 18.33
  },
  "performance": {
    "tokens_per_second": 312.5,
    "prefill_latency": 0.97,
    "token_latency": 0.00046,
    "time_for_1000_tokens": 0.46,
    "throughput_by_gpu": {
      "a100-80gb": 312.5,
      "h100-80gb": 632.8
      // Other GPUs...
    }
  },
  "overheads_used": {
    "weights": 1.05,
    "kv_cache": 1.05,
    "activations": 1.1,
    "system": 1.05
  }
}
```

### 2. `analyze_model_on_gpu` Response Format

```json
{
  "model_info": {
    "name": "model_name",
    "family": "Model Family",
    "parameters_b": 7.0,
    "hidden_dimensions": 4096,
    "feedforward_dimensions": 11008,
    "num_layers": 32,
    "vocab_size": 32000,
    "description": "Model description"
  },
  "gpu_info": {
    "name": "a100-80gb",
    "family": "NVIDIA",
    "vram_gb": 80,
    "tflops": 312.0,
    "supported_precisions": ["fp32", "fp16", "bf16"]
  },
  "analysis_parameters": {
    "sequence_length": 2048,
    "batch_size": 1,
    "precision": "fp16",
    "efficiency_factor": 0.3
  },
  "flops": {
    "attention": 171798691840,
    "feedforward": 962072674304,
    "prefill_total": 90709709291520,
    "per_token": 42950328320
  },
  "vram": {
    "model": {
      "weights_base": 13.82,
      "kv_cache_base": 0.95,
      "activations_base": 1.82,
      "weights_with_overhead": 14.51,
      "kv_cache_with_overhead": 1.0,
      "activations_with_overhead": 2.0,
      "component_subtotal": 17.51,
      "total": 18.33
    },
    "model_weights": 14.51,
    "kv_cache": 1.0,
    "activations": 2.0,
    "total": 18.33
  },
  "performance": {
    "tokens_per_second": 312.5,
    "prefill_latency": 0.97,
    "token_latency": 0.00046,
    "time_for_1000_tokens": 0.46
  },
  "compatibility": {
    "fits_on_gpu": true,
    "vram_utilization_pct": 22.91,
    "headroom_gb": 61.67,
    "minimum_required_vram_gb": 18.33
  },
  "overheads_used": {
    "weights": 1.05,
    "kv_cache": 1.05,
    "activations": 1.1,
    "system": 1.05
  }
}
```

### 3. Web UI Expected Format

The web UI expects a standardized format that combines elements from both responses:

```json
{
  "model_name": "model_name",
  "parameters_billions": 7.0,
  "vram": {
    "model_base": 13.82,
    "kv_cache_base": 0.95,
    "activations_base": 1.82,
    "model_with_overhead": 14.51,
    "kv_cache_with_overhead": 1.0,
    "activations_with_overhead": 2.0,
    "total_base": 16.59,
    "total_with_component_overhead": 17.51,
    "total_system_wide": 18.33,
    "model_per_gpu": 13.82,
    "kv_cache_per_gpu": 0.95,
    "activations_per_gpu": 1.82,
    "model_per_gpu_with_overhead": 14.51,
    "kv_cache_per_gpu_with_overhead": 1.0,
    "activations_per_gpu_with_overhead": 2.0,
    "total_base_per_gpu": 16.59,
    "total_per_gpu_with_component_overhead": 17.51
  },
  "flops": {
    "attention": 171798691840,
    "feedforward": 962072674304,
    "prefill_total": 90709709291520,
    "per_token": 42950328320
  },
  "performance": {
    "tokens_per_second": 312.5,
    "prefill_latency": 0.97,
    "token_latency": 0.00046,
    "time_for_1000_tokens": 0.46,
    "throughput_by_gpu": {
      "a100-80gb": 312.5,
      "h100-80gb": 632.8
    }
  },
  "parallelism": {
    "strategy": "none",
    "tp_size": 1,
    "pp_size": 1,
    "num_gpus": 1,
    "effective_tflops": 312.0
  },
  "overheads_used": {
    "weights": 1.05,
    "kv_cache": 1.05,
    "activations": 1.1,
    "system": 1.05
  },
  "history": ["Calculation steps..."]
}
```

## Adapter Functions

To ensure consistent responses, the web application implements adapter functions that transform the calculator responses into the expected format. These adaptations include:

1. Providing uniform field names (`prefill_total` vs `prefill`)
2. Handling different VRAM data structures
3. Adding missing fields with sensible defaults
4. Normalizing parallel processing information
5. Ensuring overheads are consistently represented
