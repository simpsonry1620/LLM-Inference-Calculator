# LLM Infrastructure Scaling Calculator - Project Structure

This document provides a detailed overview of the project structure, components, and usage for the LLM Infrastructure Scaling Calculator.

## Overview

The LLM Infrastructure Scaling Calculator is a Python toolkit designed to help estimate and plan infrastructure requirements for Large Language Models (LLMs). It provides utilities for calculating:

- VRAM requirements for model weights and inference
- Computational requirements (FLOPs) for different transformer components
- Inference throughput and latency estimates
- Multi-GPU scaling strategies
- Comparison of different hardware configurations

The project offers command-line tools, a web-based GUI, and a desktop GUI for interactive exploration.

## Directory Structure

```
LLM-Inference-Calculator/
├── docs/                      # Documentation
│   ├── project_structure.md   # This document
│   ├── advanced_calculator_usage.md # AdvancedCalculator class usage documentation
│   ├── calculator_api_description.md # API endpoint documentation
│   └── ...                    # Additional documentation
├── examples/                  # Example scripts demonstrating usage
│   ├── advanced_calculator_example.py
│   └── model_comparison_example.py
├── logs/                      # Directory for storing calculation logs
├── logging/                   # Directory for calculation history
├── src/                       # Source code
│   ├── __init__.py
│   ├── __main__.py            # Entry point for running as a module (`python -m src`)
│   └── advanced_calculator/   # Advanced calculator implementation
│       ├── __init__.py
│       ├── cli.py             # Command-line interface
│       ├── main.py            # Core calculator integration layer
│       ├── run_web.py         # Script to run the web interface
│       ├── README.md          # Documentation for the calculator
│       ├── documentation/     # Additional specific documentation (e.g., design docs)
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
│           ├── web_app.py     # Flask web application
│           └── static/        # Static files (CSS, JS) for the web app
│           └── server_logs.txt # Server-specific logs (if used)
│   └── gui/                   # Desktop GUI implementation
│       └── main.py            # Tkinter GUI application
├── templates/                 # HTML templates for the web interface
│   ├── index.html
│   └── visualize.html
├── tests/                     # Test suite
│   ├── __init__.py
│   └── test_latency.py
├── .dockerignore
├── .gitignore
├── Dockerfile                 # Docker configuration
├── README.md                  # Main project documentation
├── README_WEB_GUI.md          # Web GUI specific documentation
├── requirements.txt           # Python dependencies
├── setup.py                   # Package installation configuration
├── server_log.txt             # Server log file (specific usage TBD)
└── tatus                      # Unknown file (purpose TBD)
```

*(Note: Standard directories like `.git`, `.venv`, `.cursor`, and `__pycache__/` are omitted for clarity.)*

## Core Components

### Advanced Calculator (`src/advanced_calculator/main.py`)

The main calculator serves as an integration layer that initializes and coordinates the specialized calculator modules:

- Provides a unified API for all calculator functionalities
- Delegates calculations to specialized modules
- Maintains a history of calculations
- Provides methods for working with predefined models and GPUs
- Implements high-level analysis functions that combine multiple calculations

**For comprehensive documentation on using the AdvancedCalculator class directly in your Python code, see [Advanced Calculator Usage Guide](advanced_calculator_usage.md).**

Basic usage example:
```python
from src.advanced_calculator import AdvancedCalculator

# Initialize the calculator
calc = AdvancedCalculator()

# Calculate VRAM requirements for model weights
model_vram = calc.calculate_model_vram(
    hidden_dimensions=4096,
    feedforward_dimensions=16384,
    num_layers=32,
    vocab_size=128000,
    precision="fp16"
)

# Analyze a predefined model on a specific GPU
analysis = calc.analyze_model_on_gpu(
    model_name="llama2-7b",
    gpu_name="a100-80gb",
    sequence_length=4096,
    batch_size=1,
    precision="fp16"
)
```

### Specialized Modules (`src/advanced_calculator/modules/`)

The calculator functionality is organized into specialized modules that implement specific calculations:

#### FLOPs Calculator (`flops.py`)
- Provides methods for calculating computational requirements
- `calculate_attention`: FLOPs for attention mechanism
- `calculate_feedforward`: FLOPs for feedforward networks
- `calculate_prefill`: FLOPs for the prefill phase
- `calculate_flops_per_token`: FLOPs for generating a single token

#### VRAM Calculator (`vram.py`)
- Handles memory requirement calculations
- `calculate_model_vram`: VRAM for model weights
- `calculate_kv_cache_vram`: VRAM for KV cache
- `calculate_activations_vram`: VRAM for activations
- `calculate_total_vram`: Total VRAM with all components and overheads
- `determine_model_scaling`: Multi-GPU scaling strategies

#### Throughput Calculator (`throughput.py`)
- Estimates inference throughput
- `estimate_inference_throughput`: Tokens per second
- `estimate_batch_throughput`: Batch processing performance
- `calculate_throughput_across_gpus`: Comparison across GPU models

#### Latency Calculator (`latency.py`)
- Handles latency estimations
- `calculate_prefill_latency`: Time for processing prompt
- `calculate_token_latency`: Time for generating each token
- `calculate_completion_latency`: End-to-end completion time

#### Models Database (`models.py`)
- Maintains predefined model configurations
- Parameters for popular model architectures (Llama, Mistral, etc.)
- Methods for retrieving and filtering models

#### GPUs Database (`gpus.py`)
- Maintains specifications for various GPUs
- Performance characteristics (TFLOPS, bandwidth)
- Memory capacities and supported precisions
- Methods for filtering and selecting GPUs

#### Utilities (`utils.py`)
- Provides helper functions
- Input validation
- History tracking
- Common calculations

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

### Desktop GUI (`src/gui/main.py`)

A desktop GUI application built using Tkinter provides a native interface for the calculator:

- Input fields for model parameters (dimensions, layers, vocab), sequence lengths, batch size, precision, GPU, and efficiency.
- Predefined model and GPU selection with automatic TFLOPS filling based on selected precision.
- Displays detailed results for VRAM breakdown, FLOPs, and performance metrics (latency, throughput).
- Shows estimated parallelism strategy (Tensor Parallelism, Pipeline Parallelism) if the selected model doesn't fit on the chosen GPU.

To run the desktop GUI:
```bash
python src/gui/main.py
```

## Command-Line Interface

### CLI (`src/advanced_calculator/cli.py`)

The CLI offers detailed calculations and options through the command line:

- `calculate`: Calculate requirements with custom parameters
- `analyze-model`: Analyze a predefined model
- `list-models`: List all available predefined models
- `list-gpus`: List all available predefined GPUs
- `analyze-model-on-gpu`: Analyze a model on a specific GPU
- `recommend-gpus`: Get GPU recommendations for a model
- `filter-gpus`: Filter GPUs by criteria

To use the CLI:
```bash
# Install the package
pip install -e .

# Use the command-line tools
advanced-calculator list-models
advanced-calculator analyze-model --model-name llama2-7b
```

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

- `advanced_calculator_example.py`: Examples showcasing the calculator features
- `model_comparison_example.py`: Demonstrates how to compare different model architectures

## Tests

The `tests/` directory contains the test suite for verifying calculator functionality:

- `test_latency.py`: Tests for latency calculations

## Templates

The `templates/` directory contains HTML templates for the web interface:

- `index.html`: Main calculator interface
- `visualize.html`: Interface for visualizing and comparing saved calculations

## Documentation

- `README.md`: Main project documentation with overview and setup instructions
- `README_WEB_GUI.md`: Detailed documentation on using the web interface
- `docs/`: Additional documentation on various aspects of the project
- `docs/advanced_calculator_usage.md`: Comprehensive documentation for using the AdvancedCalculator class directly in Python code
- `docs/calculator_api_description.md`: API documentation for the calculator endpoints. **Important**: Developers should refer to this document when making API calls to ensure correct parameter usage and response handling.
