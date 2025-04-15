# LLM Scaling Calculator Web GUI

This web interface provides a user-friendly way to estimate computational requirements for large language models.

## Features

- Select from predefined model configurations or create a custom model
- Configure hardware parameters (GPU model, efficiency, parallelism strategy)
- Choose appropriate precision formats (FP16, BF16, FP32) with automatic compatibility filtering
- Visualize memory requirements, computational requirements, and performance estimates
- **NEW**: Visualize and compare past calculation results from logs
- **NEW**: Interactive table for calculation history with sorting and filtering
- Responsive design that works on desktop and mobile devices

## Setup

1. Make sure you have installed the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the web application from the project root directory:
   ```
   python -m src.advanced_calculator.run_web
   ```

3. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

## Usage

1. **Model Configuration**:
   - Select a model preset from organized family groups (Llama, Mistral, Phi, etc.)
   - Or configure a custom model by setting:
     - Hidden dimensions
     - Feedforward dimensions
     - Number of layers
     - Vocabulary size
     - Sequence length

2. **Inference Parameters**:
   - Set sequence length for context window
   - Configure batch size
   - Select precision format (FP16, BF16, FP32)

3. **Hardware Configuration**:
   - Choose a GPU model from categorized families
   - Set the efficiency factor (typically 20-30%)
   - Precision options automatically adjust based on GPU capabilities

4. Click **Calculate** to generate results

5. **NEW**: Navigate to the **Visualization Page** (via the button on the main page or by going to `/visualize`) to:
   - View a history of calculations performed through the Web UI and CLI.
   - Sort and filter the history table by various parameters (Timestamp, Model, GPU, VRAM, etc.).
   - Select calculations from the table.
   - Click "Plot Selected" to generate a bar chart comparing VRAM usage and Tokens/sec for the selected runs.
   - Hover over chart bars to see detailed configuration tooltips.

## Results

The calculator displays comprehensive results in four categories:

### Model Summary
- Model name/family
- Total parameter count in billions

### Memory Requirements
- Model weights (with 5% overhead)
- KV cache size (with 5% overhead)
- Activations memory (with 10% overhead)
- System overhead (5%)
- Total VRAM requirements

### Computational Requirements
- Attention mechanism FLOPs
- Feedforward network FLOPs
- Prefill phase (full context) FLOPs
- Per-token generation FLOPs

### Performance Estimates
- Token generation speed (tokens/second)
- Prefill latency (time to process full context)
- Per-token latency (time for each new token)
- Time to generate 1000 tokens

## Implementation Details

### Backend
- Flask web server (src/calculator_app/web_app.py)
- Uses the AdvancedCalculator class from src/calculator_app/calculator.py for computations
- Dynamically loads model configurations and GPU data when available
- RESTful API endpoints for calculation and model information

### Frontend
- Pure HTML/CSS/JavaScript (no external libraries)
- Responsive design with mobile support
- Formatted display of numeric results with appropriate units
- Dynamic precision options based on GPU capabilities

### API Endpoints
- `/api/calculate` - POST endpoint to calculate metrics
- `/api/available_models` - GET endpoint for model presets
- `/api/gpu_configs` - GET endpoint for GPU configurations
- `/api/model_config/<model_name>` - GET endpoint for specific model configuration
- `/visualize` - GET endpoint to render the history and visualization page.

## Potential Enhancements

Future improvements could include:

1. **Visualizations**:
   - Add charts/graphs to visualize scaling relationships (Partially done on `/visualize` page)
   - Compare multiple models side-by-side (Possible via `/visualize` page)

2. **Model Management**:
   - Save and load custom model configurations
   - Export results to CSV or PDF

3. **Hardware Comparison**:
   - Compare performance across multiple GPU types
   - Calculate multi-GPU scaling

4. **Cost Estimation**:
   - Integrate cloud provider pricing
   - Estimate training and inference costs

5. **UI Improvements**:
   - Add dark mode
   - Create more detailed tooltips/help information

## How It Works

The web interface communicates with the `AdvancedCalculator` class to perform the necessary calculations. The calculator follows these steps:

1. **Model Configuration Processing**:
   - Loads predefined model if selected, or uses custom parameters
   - Calculates total parameters based on dimensions

2. **Hardware Configuration**:
   - Maps selected GPU to performance characteristics
   - Applies efficiency factor to theoretical peak performance
   - Ensures precision compatibility

3. **Calculation Pipeline**:
   - Computes FLOPs for attention and feedforward components
   - Calculates memory requirements for weights, KV cache, and activations
   - Estimates throughput and latency based on computational needs and hardware capabilities

4. **Results Formatting**:
   - Organizes results into structured categories
   - Applies appropriate scaling and units (GB, TFLOPs, etc.)
   - Formats time values based on magnitude (ms, s, min, etc.)

### Logging

- Calculations run via the Web GUI (`src.advanced_calculator.web.web_app`) are logged to `logging/web_app_calculations.log`.
- This log file includes input parameters and the full JSON results.
- The `/visualize` page reads this file (and the CLI log file) to populate the history table.

1. **Visualizations**:
   - Add charts/graphs to visualize scaling relationships (Partially done on `/visualize` page)
   - Compare multiple models side-by-side (Possible via `/visualize` page)

# Optional arguments
python -m src.advanced_calculator.run_web --host 0.0.0.0 --port 8080 --debug

Available arguments:
- --host: Host address to bind to (default: 127.0.0.1)
- --port: Port to bind to (default: 5000)
- --debug: Run in debug mode for development 