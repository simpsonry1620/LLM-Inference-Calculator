# Web UI Design Documentation

This document outlines the design of the LLM Scaling Calculator web user interface.

## Overview

The web UI provides an interactive way to estimate the computational and memory requirements for large language models based on user-defined parameters or presets. It consists of an input panel for configuration and a results panel to display the calculated estimates. The interface is built using Flask and renders HTML templates (`templates/index.html` and `templates/visualize.html`). Data is processed by the `src/advanced_calculator/web/web_app.py` Flask application, which utilizes the `AdvancedCalculator` class from `src/advanced_calculator/main.py`.

For a broader understanding of the project's layout and components, refer to the [Project Structure Documentation](project_structure.md).

## Main Calculator Page (`index.html`)

This is the primary interface for performing calculations.

### 1. Input Panel

This panel contains all the configuration options for the calculation.

#### 1.1. Model Parameters Section

Defines the architecture of the language model.

| Component         | Label                    | Type            | Default Value | Description                                      | Data Source                                          |
|-------------------|--------------------------|-----------------|---------------|--------------------------------------------------|------------------------------------------------------|
| `model-preset`    | Model Preset             | Dropdown        | `custom`      | Select a predefined model or choose custom.      | User selection / List from `modules/models.py`       |
| `hidden-dim`      | Hidden Dimensions        | Number (Integer)| `768`         | Size of the model's hidden layers.               | User input / Preset value from `modules/models.py`   |
| `ff-dim`          | Feedforward Dimensions   | Number (Integer)| `3072`        | Size of the feedforward layers.                  | User input / Preset value from `modules/models.py`   |
| `num-layers`      | Number of Layers         | Number (Integer)| `12`          | Number of transformer layers in the model.       | User input / Preset value from `modules/models.py`   |
| `vocab-size`      | Vocabulary Size          | Number (Integer)| `50257`       | Number of unique tokens in the model's vocabulary.| User input / Preset value from `modules/models.py`   |

#### 1.2. Inference Parameters Section

Defines the parameters used during the inference process.

| Component      | Label            | Type            | Default Value | Description                                       | Data Source                                        |
|----------------|------------------|-----------------|---------------|---------------------------------------------------|----------------------------------------------------|
| `seq-len`      | Sequence Length  | Number (Integer)| `2048`        | Maximum sequence length the model handles.        | User input / Preset value from `modules/models.py` |
| `batch-size`   | Batch Size       | Number (Integer)| `1`           | Number of sequences processed in parallel.        | User input                                         |
| `precision`    | Precision        | Dropdown        | `fp16`        | Data type used for model weights and computation. | User selection (Options: fp16, int8, etc.)       |

#### 1.3. GPU Selection Section

Defines the hardware configuration for the calculation.

| Component          | Label                 | Type            | Default Value | Description                                      | Data Source                                       |
|--------------------|-----------------------|-----------------|---------------|--------------------------------------------------|---------------------------------------------------|
| `gpu-preset`       | GPU Preset            | Dropdown        | `a100-80gb`   | Select a predefined GPU or choose custom.        | User selection / List from `modules/gpus.py`    |
| `gpu-tflops`       | GPU TFLOPS            | Number (Float)  | `312`         | Theoretical peak performance (relevant precision). | User input / Preset value from `modules/gpus.py`  |
| `gpu-memory`       | GPU Memory (GB)       | Number (Float)  | `80`          | Available VRAM per GPU.                          | User input / Preset value from `modules/gpus.py`  |
| `gpu-mem-bandwidth`| GPU Memory Bandwidth (GB/s)| Number (Float)  | `1935`        | Speed of data transfer to/from GPU memory.       | User input / Preset value from `modules/gpus.py`  |

#### 1.4. Parallelism Section

Defines how the model is distributed across multiple GPUs.

| Component      | Label            | Type            | Default Value | Description                                     | Data Source      |
|----------------|------------------|-----------------|---------------|-------------------------------------------------|------------------|
| `parallelism-strategy` | Strategy         | Dropdown        | `None`        | Method used for distributing the model.         | User selection   |
| `tp-size`      | TP Size          | Number (Integer)| `1`           | Tensor Parallelism degree.                      | User input       |
| `pp-size`      | PP Size          | Number (Integer)| `1`           | Pipeline Parallelism degree.                    | User input       |
| `num-gpus`     | Number of GPUs   | Number (Integer)| `1`           | Total GPUs used (TP Size * PP Size).            | User input       |

#### 1.5. Calculate Button

| Component | Label     | Type   | Description                                            | Action                                                   |
|-----------|-----------|--------|--------------------------------------------------------|----------------------------------------------------------|
| `calculate-button` | Calculate | Button | Submits the form data to trigger the calculation. | Sends POST request to `/api/calculate` in `web_app.py` |

### 2. Results Panel

Displays the output of the calculations performed by the backend. Data is returned by the `/api/calculate` endpoint after processing by `AdvancedCalculator`.

#### 2.1. Model Summary Card

| Display Area           | Value Type | Description                         | Data Source (`standardize_calculator_response` output) |
|------------------------|------------|-------------------------------------|------------------------------------------------------|
| Model Name             | String     | Name of the model used.             | `model_name`                                         |
| Parameters (Billions)  | Float      | Total number of model parameters.   | `parameters_billions`                                |

#### 2.2. Computational Requirements Card

Displays estimated FLOPs (Floating Point Operations).

| Display Area             | Value Type | Description                                       | Data Source (`flops` section) |
|--------------------------|------------|---------------------------------------------------|-------------------------------|
| Attention FLOPs          | Float      | FLOPs for the attention mechanism per token.      | `attention`                   |
| Feedforward FLOPs        | Float      | FLOPs for the feedforward network per token.      | `feedforward`                 |
| **Prefill FLOPs (Total)**| Float      | Total FLOPs required for processing the prompt.   | `prefill_total`               |
| **FLOPs Per Token**      | Float      | FLOPs required to generate a single output token. | `per_token`                   |

#### 2.3. Performance Estimates Card

Displays estimated inference speed and latency.

| Display Area        | Value Type | Description                                 | Data Source (`performance` section) |
|---------------------|------------|---------------------------------------------|-----------------------------------|
| Tokens per Second   | Float      | Estimated throughput of token generation.   | `tokens_per_second`               |
| Prefill Latency (s) | Float      | Estimated time to process the input prompt. | `prefill_latency`                 |
| Token Latency (s)   | Float      | Estimated time to generate one token.       | `token_latency`                   |

#### 2.4. Parallelism Configuration Card

Displays the parallelism settings used in the calculation.

| Display Area          | Value Type | Description                          | Data Source (`parallelism` section) |
|-----------------------|------------|--------------------------------------|-------------------------------------|
| Strategy              | String     | Parallelism strategy applied.        | `strategy`                          |
| TP Size               | Integer    | Tensor Parallelism size used.        | `tp_size`                           |
| PP Size               | Integer    | Pipeline Parallelism size used.      | `pp_size`                           |
| Number of GPUs        | Integer    | Total number of GPUs used.           | `num_gpus`                          |
| Effective TFLOPS      | Float      | Combined TFLOPS across all GPUs.     | `effective_tflops`                  |

#### 2.5. Overheads Used Card

Displays the overhead factors applied during VRAM calculation.

| Display Area          | Value Type | Description                                | Data Source (`overheads_used` section) |
|-----------------------|------------|--------------------------------------------|--------------------------------------|
| Weights Overhead      | Float      | Multiplier applied to weights VRAM.        | `weights`                            |
| KV Cache Overhead     | Float      | Multiplier applied to KV cache VRAM.       | `kv_cache`                           |
| Activations Overhead  | Float      | Multiplier applied to activations VRAM.    | `activations`                        |
| System Overhead       | Float      | Multiplier applied to total component VRAM.| `system`                             |

## Visualization Page (`visualize.html`)

Accessible via a link on the main page (`/visualize`). This page is intended to display and potentially compare past calculations stored in logs. (Further details depend on the implementation of `visualize.html` and its backend route in `web_app.py`).
