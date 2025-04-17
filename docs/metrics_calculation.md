# LLM Scaling Calculator: Metrics Calculation Methodology

This document details the mathematical formulas and methodology used in the LLM Scaling Calculator for estimating computational requirements, memory usage, and performance metrics for large language models.

## 1. FLOPs Calculations

### 1.1 Attention Mechanism

The floating point operations (FLOPs) required for the attention mechanism are calculated as:

$$\text{FLOPs}_{\text{Attention}} = B \times S \times (H^2 + S \times H)$$

Where:
- $B$ = Batch size
- $S$ = Sequence length (context length)
- $H$ = Hidden dimension size

This formula accounts for:
- Query, Key, Value projection matrices: $3 \times B \times S \times H^2$
- Attention matrix computation: $B \times S \times S \times H$
- Output projection: $B \times S \times H^2$

### 1.2 Feedforward Network

The FLOPs required for the feedforward network are calculated as:

$$\text{FLOPs}_{\text{Feedforward}} = 2 \times B \times S \times H \times F$$

Where:
- $B$ = Batch size
- $S$ = Sequence length
- $H$ = Hidden dimension size
- $F$ = Feedforward dimension size

This formula accounts for:
- Up-projection: $B \times S \times H \times F$
- Down-projection: $B \times S \times F \times H$

### 1.3 Prefill Phase (Full Context Processing)

The total FLOPs for processing the entire prompt (prefill phase) are:

$$\text{FLOPs}_{\text{Prefill}} = L \times (\text{FLOPs}_{\text{Attention}} + \text{FLOPs}_{\text{Feedforward}})$$

Where:
- $L$ = Number of transformer layers

### 1.4 Per-Token Generation

The FLOPs required to generate a single new token are:

$$\text{FLOPs}_{\text{PerToken}} = L \times (\text{FLOPs}_{\text{Attention}_{\text{token}}} + \text{FLOPs}_{\text{Feedforward}_{\text{token}}})$$

Where:
- $\text{FLOPs}_{\text{Attention}_{\text{token}}} = B \times 1 \times (H^2 + S \times H)$
- $\text{FLOPs}_{\text{Feedforward}_{\text{token}}} = 2 \times B \times 1 \times H \times F$

## 2. VRAM Calculations

### 2.1 Model Weights

The VRAM required for model weights is calculated as:

$$\text{VRAM}_{\text{Weights}} = \frac{P \times b}{8 \times 1024^3} \times f_{\text{overhead}}$$

Where:
- $P$ = Total number of parameters
- $b$ = Bits per parameter (32 for FP32, 16 for FP16/BF16)
- $f_{\text{overhead}}$ = Overhead factor (typically 1.05, or 5%)

The total parameter count is calculated as:

$$P = (4 \times H^2 \times L) + (2 \times H \times F \times L) + (V \times H) + (4 \times H \times L) + (2 \times H)$$

Where:
- $H$ = Hidden dimension size
- $F$ = Feedforward dimension size
- $L$ = Number of layers
- $V$ = Vocabulary size

This includes:
- Attention layers: $4 \times H^2 \times L$ (Query, Key, Value, Output projections)
- Feedforward layers: $2 \times H \times F \times L$ (up and down projections)
- Token embeddings: $V \times H$
- Layer norm parameters: $4 \times H \times L + 2 \times H$

### 2.2 KV Cache

The KV cache VRAM is calculated as:

$$\text{VRAM}_{\text{KVCache}} = \frac{2 \times B \times S \times H \times L \times b}{8 \times 1024^3} \times f_{\text{kv\_overhead}}$$

Where:
- $B$ = Batch size
- $S$ = Sequence length
- $H$ = Hidden dimension size
- $L$ = Number of layers
- $b$ = Bits per value (32 for FP32, 16 for FP16/BF16)
- $f_{\text{kv\_overhead}}$ = KV cache overhead factor (typically 1.05, or 5%)

### 2.3 Activations

The VRAM required for activations during inference is estimated as:

$$\text{VRAM}_{\text{Activations}} = \frac{4 \times B \times S \times H \times b}{8 \times 1024^3} \times f_{\text{activations\_overhead}}$$

Where:
- $B$ = Batch size
- $S$ = Sequence length
- $H$ = Hidden dimension size
- $b$ = Bits per value (32 for FP32, 16 for FP16/BF16)
- $f_{\text{activations\_overhead}}$ = Activations overhead factor (typically 1.1, or 10%)

### 2.4 Total VRAM

The total VRAM requirement is calculated as:

$$\text{VRAM}_{\text{Total}} = (\text{VRAM}_{\text{Weights}} + \text{VRAM}_{\text{KVCache}} + \text{VRAM}_{\text{Activations}}) \times f_{\text{system\_overhead}}$$

Where:
- $f_{\text{system\_overhead}}$ = System overhead factor (typically 1.05, or 5%)

## 3. Performance Metrics

### 3.1 Throughput (Tokens per Second)

The inference throughput is calculated as:

$$\text{Tokens/Second} = \frac{\text{GPU}_{\text{FLOPS}} \times f_{\text{efficiency}}}{\text{FLOPs}_{\text{PerToken}}}$$

Where:
- $\text{GPU}_{\text{FLOPS}}$ = GPU FLOPS capacity (TFLOPs × 10¹²)
- $f_{\text{efficiency}}$ = Efficiency factor (typically 0.2-0.3, or 20-30%)
- $\text{FLOPs}_{\text{PerToken}}$ = FLOPs required per token generation

### 3.2 Prefill Latency

The time required to process the full context (prefill phase) is:

$$\text{Latency}_{\text{Prefill}} = \frac{\text{FLOPs}_{\text{Prefill}}}{\text{GPU}_{\text{FLOPS}} \times f_{\text{efficiency}}}$$

*(Note: This currently only considers compute limitations. Memory bandwidth effects on prefill are a TODO.)*

### 3.3 Token Generation Latency

The time required to generate a single token considers both compute and memory bandwidth limitations. The latency is the maximum of the compute-bound latency and the memory-bound latency.

**Compute Latency:**

$$\text{Latency}_{\text{Token (Compute)}} = \frac{\text{FLOPs}_{\text{PerToken}}}{\text{GPU}_{\text{FLOPS}} \times f_{\text{efficiency}}}$$

**Memory Latency:**
This assumes all model parameters need to be loaded from memory per token.

$$\text{Latency}_{\text{Token (Memory)}} = \frac{\text{Model Size (Bytes)}}{\text{Effective Memory Bandwidth (Bytes/s)}}$$

$$\text{Model Size (Bytes)} = P_{\text{Billion}} \times 10^9 \times b$$
$$\text{Effective Memory Bandwidth} = \text{BW}_{\text{GB/s}} \times 10^9 \times f_{\text{efficiency}}$$

Where:
- $P_{\text{Billion}}$ = Model parameters in billions
- $b$ = Bytes per parameter based on precision
- $\text{BW}_{\text{GB/s}}$ = GPU memory bandwidth in GB/s

**Final Token Latency:**

$$\text{Latency}_{\text{Token}} = \max(\text{Latency}_{\text{Token (Compute)}}, \text{Latency}_{\text{Token (Memory)}})$$

### 3.4 Time for N Tokens

The total time to generate N tokens after the prefill phase is:

$$\text{Time}_{\text{N Tokens}} = \text{Latency}_{\text{Token}} \times N$$

## 4. Precision Impact

Different precision formats affect both memory usage and computational performance:

| Precision | Bytes per Parameter | Relative Memory Usage | Relative Performance |
|-----------|---------------------|------------------------|----------------------|
| FP32      | 4                   | 100%                   | Baseline            |
| FP16/BF16 | 2                   | 50%                    | 2-4× faster         |

## 5. GPU Efficiency Considerations

The real-world performance of GPUs for LLM inference is affected by several factors:

$$f_{\text{efficiency}} = f_{\text{base}} \times f_{\text{memory\_bound}} \times f_{\text{utilization}} \times f_{\text{framework}}$$

Where:
- $f_{\text{base}}$ = Base efficiency factor (typically 0.3-0.5)
- $f_{\text{memory\_bound}}$ = Reduction due to memory bandwidth limitations (0.5-0.9)
- $f_{\text{utilization}}$ = Reduction due to suboptimal compute utilization (0.6-0.9)
- $f_{\text{framework}}$ = Reduction due to framework overhead (0.7-0.95)

In practice, we use a simplified efficiency factor that encompasses all these considerations, typically 0.2-0.3 (20-30%) of theoretical peak performance.

## 6. Multi-GPU Scaling

When distributing a model across multiple GPUs, the theoretical speedup follows:

$$\text{Speedup} = N_{\text{GPU}} \times f_{\text{scaling}}$$

Where:
- $N_{\text{GPU}}$ = Number of GPUs
- $f_{\text{scaling}}$ = Scaling efficiency factor (typically 0.7-0.9)

The scaling efficiency decreases with more GPUs due to communication overhead and depends on the parallelization strategy (tensor, pipeline, or sequence parallelism). 