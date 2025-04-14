# LLM Infrastructure Scaling Calculator

A Python-based calculator for estimating and planning Large Language Model (LLM) infrastructure requirements.

## Features

- Calculate compute resources required for LLM training and inference
- Estimate VRAM requirements for model weights and activations
- Calculate FLOPs for various transformer components
- Predict throughput and latency for inference workloads (primarily via Web GUI)
- Compare scaling options for optimal performance
- **NEW**: Predefined model architectures and GPU configurations
- **NEW**: Multi-GPU scaling strategy recommendations (Tensor Parallelism, Pipeline Parallelism)
- **Web GUI** for interactive exploration and visualization
- **NEW**: Calculation logging (CLI & Web) to `logging/` directory
- **NEW**: Web-based visualization page (`/visualize`) to view, sort, filter, and plot calculation history

## Setup

1. Clone this repository
2. Create and activate the virtual environment:
   ```
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Web GUI

The easiest way to use the calculator is through the web interface:

```
# Make sure you are in the project root directory
python -m src.calculator_app.web_app
```

Then open your browser to `http://127.0.0.1:5000/` to access the interactive calculator.

**NEW:** You can also navigate to `http://127.0.0.1:5000/visualize` (or use the button on the main page) to view and compare previous calculation results.

See [README_WEB_GUI.md](README_WEB_GUI.md) for more details.

### Advanced Calculator (Direct Usage)

While the Web GUI provides the most comprehensive interface, you can use the core calculator directly for specific calculations:

```python
# Example assuming your script is outside the src/ directory
# or src/ is in your PYTHONPATH
from src.calculator_app.calculator import AdvancedCalculator

# Initialize the advanced calculator
adv_calc = AdvancedCalculator()

# Calculate FLOPs for attention mechanism
flops_attention = adv_calc.calculate_flops_attention(
    batch_size=32,
    sequence_length=2048,
    hidden_dimensions=4096
)

# Calculate VRAM requirements for model weights
model_vram = adv_calc.calculate_model_vram(
    hidden_dimensions=4096,
    feedforward_dimensions=16384,
    num_layers=32,
    vocab_size=128000,
    precision="fp16"
)

# Calculate total VRAM including KV cache and activations
total_vram = adv_calc.calculate_total_vram(
    batch_size=1,
    sequence_length=8192,
    hidden_dimensions=8192,
    feedforward_dimensions=28672,
    num_layers=80,
    vocab_size=128000,
    precision="fp16",
    weights_overhead=1.05,
    kv_cache_overhead=1.05,
    activations_overhead=1.1,
    system_overhead=1.05
)

# Determine multi-GPU scaling strategy for a given GPU VRAM
scaling_info = adv_calc.determine_model_scaling(
    gpu_vram_gb=80.0, # Example: 80GB GPU
    batch_size=1,
    sequence_length=8192,
    hidden_dimensions=8192,
    feedforward_dimensions=28672,
    num_layers=80,
    vocab_size=128000,
    precision="fp16"
)

print(f"Attention FLOPs: {flops_attention:,}")
print(f"Model VRAM: {model_vram:.2f} GB")
print(f"Total Estimated VRAM: {total_vram:.2f} GB")
print(f"Scaling Strategy: {scaling_info['recommended_strategy']}")
print(f"Required GPUs: {scaling_info['num_gpus_required']}")

# Note: Throughput and latency calculations are more complex and typically
# handled within the Web GUI (src/calculator_app/web_app.py) which incorporates efficiency factors
# and GPU performance data. Direct calculation requires providing GPU TFLOPS
# and efficiency estimates.
```

### Using Predefined Models & GPUs

Predefined model architectures and GPU specifications are primarily utilized by the **Web GUI** (`src/calculator_app/web_app.py`). The GUI logic itself contains or imports these configurations (potentially from helper modules within `src/calculator_app/` if refactored).

Directly accessing these predefined configurations in a script would involve importing functions from `src.calculator_app.web_app` or the respective modules if needed, which is less common for typical usage.

## Calculation Approaches and Assumptions

For detailed information about the calculations and assumptions used, please see:
- [docs/assumptions.md](docs/assumptions.md) - Key assumptions and methodology overview
- [docs/metrics_calculation.md](docs/metrics_calculation.md) - Detailed mathematical formulas and equations

### FLOPs Calculations

Our FLOPs calculations are based on the following approaches:

1. **Attention Mechanism**: 
   - Formula: `Batch_Size * Sequence_Length * (Hidden_Dimensions^2 + Sequence_Length * Hidden_Dimensions)`
   - Accounts for query/key/value projections and attention matrix operations
   - Assumes standard scaled dot-product attention without optimizations

2. **Feedforward Network**:
   - Formula: `2 * Batch_Size * Sequence_Length * Hidden_Dimensions * FeedForward_Dimensions`
   - Accounts for both the up-projection and down-projection operations
   - Assumes standard MLP with two linear layers and activation in between

3. **Prefill Phase**:
   - Aggregates FLOPs across all transformer layers for both attention and feedforward operations
   - Formula: `Num_Layers * (FLOPs_Attention + FLOPs_Feedforward)`

### VRAM Calculations

VRAM estimates include:

1. **Model Weights**:
   - Accounts for attention blocks, feedforward layers, embeddings, and layer norms
   - Considers precision (fp16, bf16, fp32) for parameter size
   - Includes configurable overhead factor for framework-specific memory usage

2. **KV Cache**:
   - Stores key and value tensors for previously processed tokens
   - Size scales with batch size, sequence length, model dimensions, and layers
   - Essential for efficient autoregressive generation
   - Includes configurable overhead factor

3. **Activations**:
   - Includes temporary activations during forward pass (excluding KV cache)
   - Accounts for memory alignment and framework overhead via a configurable factor

4. **Total VRAM**:
   - Sums the above components and applies an additional system-level overhead factor.

### Throughput and Latency

1. **Throughput Estimation**:
   - Based on GPU FLOPS capacity and model's FLOPS requirements
   - Includes efficiency factor to account for real-world performance (typically 20-30% of theoretical peak)
   - Considers parallelization across multiple GPUs with scaling efficiency

2. **Latency Calculation**:
   - Separates prefill phase (processing full context) from token generation
   - Accounts for efficiency factors based on empirical observations
   - Provides time-to-first-token and per-token generation estimates

### Multi-GPU Scaling

The `determine_model_scaling` function provides a heuristic recommendation for distributing a large model across multiple GPUs using Tensor Parallelism (TP) and Pipeline Parallelism (PP).

- It calculates the total VRAM required.
- Compares required VRAM to the VRAM of a single target GPU.
- If the model doesn't fit, it estimates the minimum number of GPUs.
- It proposes a TP and PP degree (e.g., `TP=8, PP=2`) to distribute the model.
- It estimates the VRAM required per GPU under the proposed scaling strategy.
- *Note*: This is a heuristic, and optimal scaling depends heavily on the specific hardware interconnect and framework implementation.

## Key Assumptions

- **GPU Efficiency**: Real-world performance is typically 20-30% of theoretical peak FLOPS
- **Memory Overhead**: 
  - Weights typically require 5-10% extra memory due to optimizer states and framework overhead
  - KV cache may require 5% extra for memory alignment
  - Activations usually need 10-20% extra due to intermediate computations
- **Multi-GPU Scaling**: Efficiency decreases with more GPUs due to communication overhead
- **Precision Impact**: Using lower precision (fp16/bf16) reduces memory by ~50% vs fp32 but may require additional overhead for stability

## Predefined Models

The calculator (via the Web GUI) includes architecture details for many popular LLM models, including:

- **Llama Family**: Llama 2 (7B, 13B, 70B), Llama 3 (8B, 70B)
- **Mistral Family**: Mistral 7B, Mixtral 8x7B
- **Phi Family**: Phi-2, Phi-3 Mini
- **Other Examples**: Small, Medium, Large, XL defaults

The Web GUI allows selecting these models to pre-fill parameters. GPU configurations (like A100, H100, H200) are also available in the GUI.

See the `examples/` directory and the Web GUI for demonstrations.

## Project Structure

- `src/`
  - `calculator_app/` - Main Python package for the calculator.
    - `__init__.py` - Package marker.
    - `web_app.py` - Flask application for the Web GUI interface. Contains API endpoints and potentially helper modules.
    - `calculator.py` - Core calculation logic for FLOPs, VRAM, and scaling.
    - `modules/` (Optional/Potential) - Could contain helper modules for models, GPUs, utils if refactored out of `web_app.py`.
- `templates/` - HTML templates for web GUI (`index.html`, `visualize.html`).
- `static/` - Static assets (CSS, JS) for web GUI (if any added beyond CDNs).
- `tests/` - Unit and integration tests.
- `docs/` - Documentation files (`assumptions.md`, `metrics_calculation.md`).
- `examples/` - Example scripts and notebooks.
- `logging/` - Directory containing calculation log files (ignored by git).
- `requirements.txt` - Python package dependencies.
- `setup.py` - Python package setup script.
- `.gitignore` - Git ignore file.
- `README.md` - This file.
- `README_WEB_GUI.md` - Detailed information about the Web GUI.

## Web Interface

The web interface provides an intuitive way to interact with the calculator and visualize results. Features include:

- Selection of predefined model architectures
- Custom model configuration
- Selection of GPU hardware with performance specifications
- Interactive efficiency adjustment
- Detailed breakdown of memory requirements
- Computational requirements analysis
- Performance metrics and latency estimates
- Performance metrics (Throughput, Latency) and scaling recommendations
- **NEW**: Calculation history view with sorting, filtering, and plotting capabilities at `/visualize`

## Logging and Visualization (New)

- Calculations run via the CLI (`src.cli`) are logged to `logging/cli_calculations.log`.
- Calculations run via the Web GUI (`src.calculator_app.web_app`) are logged to `logging/web_app_calculations.log`.
- These logs contain input parameters and JSON results.
- The `logging/` directory is included in `.gitignore`.
- A visualization page is available in the Web GUI at the `/visualize` endpoint.
- This page displays a sortable and filterable table of logged calculations (from both CLI and Web logs).
- Users can select entries from the table and plot comparisons of VRAM usage and Tokens/sec. 