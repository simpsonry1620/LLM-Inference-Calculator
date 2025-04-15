# LLM Infrastructure Scaling Calculator - Project Structure

This document provides a detailed overview of the project structure, components, and usage for the LLM Infrastructure Scaling Calculator.

## Overview

The LLM Infrastructure Scaling Calculator is a Python toolkit designed to help estimate and plan infrastructure requirements for Large Language Models (LLMs). It provides utilities for calculating:

- VRAM requirements for model weights and inference
- Computational requirements (FLOPs) for different transformer components
- Inference throughput and latency estimates
- Multi-GPU scaling strategies
- Comparison of different hardware configurations

The project offers both command-line tools and a web-based GUI for interactive exploration.

## Directory Structure

```
LLM-Inference-Calculator/
├── docs/                      # Documentation
│   ├── project_structure.md   # This document
│   └── ...                    # Additional documentation
├── examples/                  # Example scripts demonstrating usage
│   ├── basic_usage.py
│   ├── advanced_calculator_example.py
│   └── model_comparison_example.py
├── logs/                      # Directory for storing calculation logs
├── logging/                   # Directory for calculation history
├── src/                       # Source code
│   ├── __init__.py
│   ├── __main__.py            # Entry point for running as a module
│   ├── calculator.py          # Basic calculator implementation
│   ├── cli.py                 # Command-line interface
│   └── advanced_calculator/   # Advanced calculator implementation
│       ├── __init__.py
│       ├── cli.py             # Advanced CLI
│       ├── main.py            # Core advanced calculator implementation
│       ├── run_web.py         # Script to run the web interface
│       ├── README.md          # Documentation for the advanced calculator
│       ├── modules/           # Specialized calculator modules
│       │   ├── __init__.py
│       │   ├── flops.py       # FLOPs calculations
│       │   ├── gpus.py        # GPU specifications and capabilities
│       │   ├── latency.py     # Latency estimations
│       │   ├── models.py      # Model architecture specifications
│       │   ├── throughput.py  # Throughput calculations
│       │   ├── utils.py       # Utility functions
│       │   └── vram.py        # Memory requirement calculations
│       └── web/               # Web interface implementation
│           ├── __init__.py
│           └── web_app.py     # Flask web application
├── templates/                 # HTML templates for the web interface
│   ├── index.html
│   └── visualize.html
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── test_calculator.py
│   └── test_latency.py
├── .dockerignore
├── .gitignore
├── Dockerfile                 # Docker configuration
├── README.md                  # Main project documentation
├── README_WEB_GUI.md          # Web GUI specific documentation
├── requirements.txt           # Python dependencies
└── setup.py                   # Package installation configuration
```

## Core Components

### Basic Calculator (`src/calculator.py`)

The basic calculator provides fundamental calculations for LLM infrastructure planning:

- Memory requirements estimation (`estimate_memory_requirements`)
- Compute requirements estimation (`estimate_compute_requirements`)
- Combined resource estimation (`estimate_resources`)

Usage example:
```python
from src import calculator

# Initialize with default A100 settings
calc = calculator.LLMScalingCalculator()

# Estimate resources for a 7B parameter model
results = calc.estimate_resources(
    model_size_params=7e9,  # 7 billion parameters
    batch_size=32,
    sequence_length=2048,
    tokens_to_train=300e9  # 300B tokens
)
```

### Advanced Calculator (`src/advanced_calculator/main.py`)

The advanced calculator builds on the basic functionality with detailed estimations for specific components:

- **FLOPs Calculations**:
  - Attention mechanism (`calculate_flops_attention`)
  - Feedforward networks (`calculate_flops_feedforward`)
  - Prefill phase (`calculate_flops_prefill`)

- **VRAM Calculations**:
  - Model weights (`calculate_model_vram`)
  - KV cache (`calculate_kv_cache_vram`)
  - Activations (`calculate_activations_vram`)
  - Total VRAM (`calculate_total_vram`)

- **Throughput and Latency**:
  - Inference throughput (`estimate_inference_throughput`)
  - Batch throughput (`estimate_batch_throughput`)
  - Prefill latency (`estimate_prefill_latency`)
  - Token generation latency (`estimate_token_generation_latency`)
  - Completion latency (`estimate_completion_latency`)

- **Model and GPU Database**:
  - Predefined model configurations (`get_model_config`, `get_available_models`)
  - Predefined GPU configurations (`get_gpu_config`, `get_available_gpus`)
  - Hardware recommendations (`get_recommended_gpus_for_model`)

- **Analysis Methods**:
  - Complete model analysis on specific GPU (`analyze_model_on_gpu`)
  - Generic model analysis by name (`analyze_model_by_name`)

Usage example:
```python
from src.advanced_calculator import AdvancedCalculator

# Initialize the advanced calculator
adv_calc = AdvancedCalculator()

# Calculate VRAM requirements for model weights
model_vram = adv_calc.calculate_model_vram(
    hidden_dimensions=4096,
    feedforward_dimensions=16384,
    num_layers=32,
    vocab_size=128000,
    precision="fp16"
)

# Analyze a predefined model on a specific GPU
analysis = adv_calc.analyze_model_on_gpu(
    model_name="llama2-7b",
    gpu_name="a100-80gb",
    sequence_length=4096,
    batch_size=1,
    precision="fp16"
)
```

### Specialized Modules (`src/advanced_calculator/modules/`)

The advanced calculator is organized into specialized modules:

#### FLOPs Calculator (`flops.py`)
Handles calculations for computational requirements of transformer operations.

#### VRAM Calculator (`vram.py`)
Calculates memory requirements for different components of the model during inference.

#### Throughput Calculator (`throughput.py`)
Estimates tokens per second processing capabilities.

#### Latency Calculator (`latency.py`)
Calculates time-to-first-token and per-token generation times.

#### Models Database (`models.py`)
Maintains a database of predefined model architectures with their parameters.

#### GPUs Database (`gpus.py`)
Maintains specifications and capabilities of different GPU models.

#### Utilities (`utils.py`)
Provides helper functions and history tracking for calculations.

### Web Interface (`src/advanced_calculator/web/web_app.py`)

The web interface provides an interactive GUI for exploring model calculations:

- Input forms for model parameters and hardware specifications
- Result visualization with detailed breakdowns
- Predefined model and GPU selection
- Visualization page for comparing previous calculations
- Logging of calculation history

To run the web interface:
```bash
python -m src.advanced_calculator.run_web
```

Then open your browser to `http://127.0.0.1:5000/` to access the interactive calculator.

## Command-Line Interfaces

### Basic CLI (`src/cli.py`)

The basic CLI provides command-line access to fundamental calculations.

### Advanced CLI (`src/advanced_calculator/cli.py`)

The advanced CLI offers more detailed calculations and options through the command line.

## Docker Deployment

For easy deployment, the project includes Docker configuration:

1. Build the Docker image:
   ```bash
   docker build -t llm-scaling-calculator .
   ```

2. Run the Docker container:
   ```bash
   docker run -p 5000:5000 llm-scaling-calculator
   ```

3. Access the application at `http://localhost:5000/`.

## Examples

The `examples/` directory contains sample scripts demonstrating various usage patterns:

- `basic_usage.py`: Simple examples using the base calculator
- `advanced_calculator_example.py`: Examples showcasing the advanced calculator features
- `model_comparison_example.py`: Demonstrates how to compare different model architectures

## Tests

The `tests/` directory contains the test suite for verifying calculator functionality:

- `test_calculator.py`: Tests for the basic calculator functionality
- `test_latency.py`: Tests for latency calculations

## Templates

The `templates/` directory contains HTML templates for the web interface:

- `index.html`: Main calculator interface
- `visualize.html`: Interface for visualizing and comparing saved calculations

## Documentation

- `README.md`: Main project documentation with overview and setup instructions
- `README_WEB_GUI.md`: Detailed documentation on using the web interface
- `docs/`: Additional documentation on various aspects of the project
