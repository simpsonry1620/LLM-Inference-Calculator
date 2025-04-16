# Calculator API Response Format

This document outlines the standardized response format for the LLM Inference Calculator API.

## Overview

The calculator's unified response format includes the following major sections:
- Model information
- VRAM requirements
- Computational requirements (FLOPs)
- Performance estimates
- Parallelism configuration
- Overhead factors used
- Calculation history

## Response Schema

```json
{
  "model_name": "string",
  "parameters_billions": "number",
  "error": "string (optional)",
  "vram": {
    "model_base": "number (bytes)",
    "model_with_overhead": "number (bytes)",
    "model_per_gpu": "number (bytes)",
    "model_per_gpu_with_overhead": "number (bytes)",
    "kv_cache_base": "number (bytes)",
    "kv_cache_with_overhead": "number (bytes)",
    "kv_cache_per_gpu": "number (bytes)",
    "kv_cache_per_gpu_with_overhead": "number (bytes)",
    "activations_base": "number (bytes)",
    "activations_with_overhead": "number (bytes)",
    "activations_per_gpu": "number (bytes)",
    "activations_per_gpu_with_overhead": "number (bytes)",
    "total_base": "number (bytes)",
    "total_with_component_overhead": "number (bytes)",
    "total_base_per_gpu": "number (bytes)",
    "total_per_gpu_with_component_overhead": "number (bytes)",
    "total_system_wide": "number (bytes)"
  },
  "flops": {
    "attention": "number",
    "feedforward": "number",
    "prefill_total": "number",
    "per_token": "number"
  },
  "performance": {
    "prefill_latency": "number (seconds)",
    "token_latency": "number (seconds)",
    "tokens_per_second": "number"
  },
  "parallelism": {
    "strategy": "string ('none', 'tensor', 'pipeline', 'tensor_pipeline')",
    "num_gpus": "number",
    "tp_size": "number",
    "pp_size": "number",
    "effective_tflops": "number"
  },
  "overheads_used": {
    "weights": "number",
    "kv_cache": "number",
    "activations": "number",
    "system": "number"
  },
  "history": ["string array of calculation steps"]
}
```

## Field Details

### Model Information
- `model_name`: Name of the model being analyzed or "custom" for user-defined models
- `parameters_billions`: Total model parameters in billions
- `error`: Error message if calculation failed (only present if there's an error)

### VRAM Requirements (`vram` object)
- `model_base`: Base memory required for model weights without overhead
- `model_with_overhead`: Memory required for model weights with component overhead
- `model_per_gpu`: Per-GPU base memory requirement for model weights (without overhead)
- `model_per_gpu_with_overhead`: Per-GPU memory requirement for model weights with component overhead
- `kv_cache_base`: Base memory required for KV cache without overhead
- `kv_cache_with_overhead`: Memory required for KV cache with component overhead
- `kv_cache_per_gpu`: Per-GPU base memory requirement for KV cache (without overhead)
- `kv_cache_per_gpu_with_overhead`: Per-GPU memory requirement for KV cache with component overhead
- `activations_base`: Base memory required for activations without overhead
- `activations_with_overhead`: Memory required for activations with component overhead
- `activations_per_gpu`: Per-GPU base memory requirement for activations (without overhead)
- `activations_per_gpu_with_overhead`: Per-GPU memory requirement for activations with component overhead
- `total_base`: Total VRAM requirement without any overhead
- `total_with_component_overhead`: Total VRAM with per-component overhead applied
- `total_base_per_gpu`: Per-GPU total base VRAM requirement (without overhead)
- `total_per_gpu_with_component_overhead`: Per-GPU total VRAM with per-component overhead applied
- `total_system_wide`: Total system-wide VRAM requirement (includes system-wide overhead)

### Computational Requirements (`flops` object)
- `attention`: FLOPs required for attention mechanism
- `feedforward`: FLOPs required for feedforward network
- `prefill_total`: Total FLOPs for prefill phase
- `per_token`: FLOPs required per token generation

### Performance Estimates (`performance` object)
- `prefill_latency`: Estimated time for prefill phase in seconds
- `token_latency`: Estimated time per token generation in seconds
- `tokens_per_second`: Estimated tokens generated per second

### Parallelism Configuration (`parallelism` object)
- `strategy`: Parallelism strategy used ("none", "tensor", "pipeline", "tensor_pipeline")
- `num_gpus`: Total number of GPUs
- `tp_size`: Tensor parallelism size
- `pp_size`: Pipeline parallelism size
- `effective_tflops`: Effective TFLOPS capability of the GPU configuration

### Overhead Factors (`overheads_used` object)
- `weights`: Overhead factor applied to model weights
- `kv_cache`: Overhead factor applied to KV cache
- `activations`: Overhead factor applied to activations
- `system`: System-wide overhead factor

### Calculation History
- `history`: Array of strings describing calculation steps

## Example Response

```json
{
  "model_name": "llama-7b",
  "parameters_billions": 7.24,
  "vram": {
    "model_base": 14480000000,
    "model_with_overhead": 15204000000,
    "kv_cache_base": 536870912,
    "kv_cache_with_overhead": 563714458,
    "activations_base": 268435456,
    "activations_with_overhead": 295279002,
    "total_base": 15285306368,
    "total_with_component_overhead": 16062993460,
    "total_system_wide": 16866143133
  },
  "flops": {
    "attention": 4294967296,
    "feedforward": 12884901888,
    "prefill_total": 35115649277952,
    "per_token": 17179869184
  },
  "performance": {
    "prefill_latency": 0.85,
    "token_latency": 0.04,
    "tokens_per_second": 25
  },
  "parallelism": {
    "strategy": "none",
    "num_gpus": 1,
    "tp_size": 1,
    "pp_size": 1,
    "effective_tflops": 42
  },
  "overheads_used": {
    "weights": 1.05,
    "kv_cache": 1.05,
    "activations": 1.10,
    "system": 1.05
  },
  "history": [
    "Calculating model parameters: 7.24B",
    "Calculating VRAM for weights: 14.48GB",
    "Calculating KV cache: 0.54GB",
    "Applying overheads to memory components",
    "Calculating total VRAM requirement: 16.87GB"
  ]
}
``` 