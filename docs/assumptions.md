# LLM Scaling Assumptions and Methodology

This document outlines the key assumptions and methodology used in our LLM scaling calculator.

## FLOPs Calculations

### Attention Mechanism

For the attention mechanism, we calculate FLOPs using the formula:
```
FLOPs_Attention = Batch_Size * Sequence_Length * (Hidden_Dimensions^2 + Sequence_Length * Hidden_Dimensions)
```

**Key Assumptions**:
- This accounts for QKV projections (3 matrix multiplications), the attention score calculation, and the output projection.
- We assume standard scaled dot-product attention without optimizations like sparse attention or multi-query attention.
- Each token attends to all tokens in the context (full attention), which is quadratic with sequence length.
- We don't account for potential optimizations like FlashAttention or memory-efficient attention algorithms.

### Feedforward Network

For the feedforward network, we calculate FLOPs using the formula:
```
FLOPs_Feedforward = 2 * Batch_Size * Sequence_Length * Hidden_Dimensions * FeedForward_Dimensions
```

**Key Assumptions**:
- The feedforward network consists of two fully connected layers with an activation function in between.
- The first layer expands from `Hidden_Dimensions` to `FeedForward_Dimensions`.
- The second layer projects back from `FeedForward_Dimensions` to `Hidden_Dimensions`.
- We don't include the FLOPs required for activation functions (like GELU, SiLU, etc.), as they are relatively small.
- Typically, `FeedForward_Dimensions` is 4x the `Hidden_Dimensions`, but this can vary by architecture.

### Prefill vs. Generation

**Prefill Phase**:
- During prefill, the model processes the entire prompt all at once.
- All tokens attend to all tokens that came before them.
- FLOPs are calculated for the full sequence length.

**Generation Phase**:
- During token generation, only a single new token is generated at a time.
- This new token attends to all previous tokens (including the prompt).
- The KV cache stores the key and value tensors from previous tokens to avoid recomputation.
- The per-token FLOPs are significantly less than processing the entire sequence.

## VRAM Calculations

### Model Weights

**Key Assumptions**:
- Each parameter requires different bytes based on precision:
  - FP32: 4 bytes per parameter
  - FP16/BF16: 2 bytes per parameter
- The model has these main components:
  - Attention layers: Query, Key, Value, and Output projections (4 matrices)
  - Feedforward layers: Up-projection and down-projection (2 matrices)
  - Layer normalization: Parameters for each layer norm
  - Token embeddings: A large embedding table
- We include an overhead factor to account for extra memory used by the framework.

### KV Cache

The KV cache is the memory required to store key and value tensors for previously processed tokens.

**Key Assumptions**:
- For each layer, we store both key and value tensors.
- These tensors have shape `[batch_size, sequence_length, hidden_dimensions]`.
- The memory requirement is calculated as:
  ```
  KV_Cache_Bytes = 2 * batch_size * sequence_length * hidden_dimensions * num_layers * bytes_per_value
  ```
- Long context models require larger KV caches, which can be a significant memory bottleneck.

### Activations

**Key Assumptions**:
- Activations include temporary tensors created during inference.
- The major components are the KV cache and intermediate activations during computation.
- We apply a higher overhead factor to activations (10-20%) due to additional temporary buffers.
- The activations memory scales linearly with batch size and sequence length.

## Performance Considerations

### GPU Efficiency

**Key Assumptions**:
- Real-world performance is typically 20-30% of theoretical peak FLOPS.
- Factors reducing efficiency include:
  - Memory bandwidth limitations
  - Kernel launch overhead
  - Suboptimal compute utilization
  - Framework overhead
  - Parallel communication overhead

### GPU Hardware

The calculator supports a variety of GPU hardware types:

- **NVIDIA Datacenter GPUs**: A100, H100, H200
- **NVIDIA Consumer GPUs**: RTX 30-series, RTX 40-series
- **Other Hardware**: Various accelerator types

**Key Assumptions**:
- Different GPU models have different performance characteristics for different precision types
- Performance varies by:
  - FP16/BF16/FP32 support and throughput
  - VRAM capacity
  - Architecture generation
- Newer GPUs may support specialized formats like FP8 or have tensor cores optimized for certain operations
- The calculator maps GPU hardware to appropriate performance metrics for calculations

### Precision Types

The calculator supports multiple precision types:

- **FP32 (32-bit floating-point)**:
  - Highest precision, requires 4 bytes per parameter
  - Lower throughput compared to reduced precision formats
  - May be required for certain operations or for model stability

- **FP16 (16-bit floating-point)**:
  - Half-precision, requires 2 bytes per parameter
  - Higher throughput than FP32
  - May suffer from numerical instability for certain operations

- **BF16 (brain floating-point)**:
  - Alternative 16-bit format with better numerical stability than FP16
  - Same memory requirements as FP16 (2 bytes per parameter)
  - Provides a balance between precision and performance

**Key Assumptions**:
- The model can operate effectively in the selected precision
- The selected GPU hardware supports the chosen precision format
- We automatically filter GPU-precision combinations that aren't compatible

### Scaling Across GPUs

**Key Assumptions**:
- Scaling efficiency decreases with more GPUs due to communication overhead.
- Different parallelism strategies (tensor, pipeline, sequence) have different efficiency characteristics.
- The calculation uses a simplified scaling model that applies an efficiency factor to multi-GPU setups.

### Latency Components

**Key Assumptions**:
- Prefill latency depends on the entire prompt length and scales with sequence length.
- Per-token generation latency is much smaller but consistent for each new token.
- Total completion latency = Prefill Latency + (Generation Latency * Output Length)
- The time to first token (TTFT) is dominated by the prefill latency.

## Overall Methodology

Our calculator uses these formulas and assumptions to provide estimates for:
1. Computational requirements (FLOPs)
2. Memory requirements (VRAM)
3. Throughput (tokens per second)
4. Latency (seconds for prefill and per-token generation)

These estimates should be treated as approximations rather than exact values, as real-world performance can vary significantly based on hardware, software implementation, and specific model architecture details. 