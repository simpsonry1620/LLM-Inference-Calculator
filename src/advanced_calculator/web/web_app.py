#!/usr/bin/env python3
import sys
import os
import logging
import json
import re
import datetime
import traceback
from flask import Flask, render_template, request, jsonify

from src.advanced_calculator.main import AdvancedCalculator
from src.advanced_calculator.modules.gpus import (
    get_gpu_config,
    get_gpu_families,
    list_all_gpus
)
from src.advanced_calculator.modules.models import (
    get_model_config,
    get_model_families,
    list_all_models
)

def create_app():
    """Create and configure the Flask application."""
    # Initialize Flask, specifying template and static folder locations
    app = Flask(__name__, 
                template_folder='../../../templates', 
                static_folder='../../../static')

    # Setup logging for the web app
    log_dir = "logs/web"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    web_log_file = os.path.join(log_dir, "web_app_calculations.log")

    # Use Flask's built-in logger
    file_handler = logging.FileHandler(web_log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)

    # Initialize calculator
    calculator = AdvancedCalculator()

    def standardize_calculator_response(analysis, data):
        """
        Standardize calculator responses to a consistent format expected by the web UI.
        
        This function provides a single point of standardization for all calculator responses,
        handling both analyze_model_by_name and analyze_model_on_gpu output formats.
        
        Args:
            analysis: Raw calculator response (from analyze_model_by_name or analyze_model_on_gpu)
            data: Original request data
            
        Returns:
            Standardized analysis data in the format expected by the web UI with the following structure:
            {
                'model_name': str,                    # Name of the model analyzed
                'parameters_billions': float,         # Model parameter count in billions
                'vram': {                             # Memory requirements
                    'model_base': float,              # Base model weights memory without overhead
                    'kv_cache_base': float,           # Base KV cache memory without overhead
                    'activations_base': float,        # Base activations memory without overhead
                    'model_with_overhead': float,     # Model weights memory with overhead
                    'kv_cache_with_overhead': float,  # KV cache memory with overhead
                    'activations_with_overhead': float, # Activations memory with overhead
                    'total_base': float,              # Total memory without overhead
                    'total_with_component_overhead': float, # Total with per-component overhead
                    'total_system_wide': float,       # Total with system-wide overhead
                    'model_per_gpu': float,           # Model weights per GPU (for multi-GPU)
                    'kv_cache_per_gpu': float,        # KV cache per GPU (for multi-GPU)
                    'activations_per_gpu': float,     # Activations per GPU (for multi-GPU)
                    'model_per_gpu_with_overhead': float, # Model weights per GPU with overhead
                    'kv_cache_per_gpu_with_overhead': float, # KV cache per GPU with overhead
                    'activations_per_gpu_with_overhead': float, # Activations per GPU with overhead
                    'total_base_per_gpu': float,      # Total per GPU without overhead
                    'total_per_gpu_with_component_overhead': float # Total per GPU with overhead
                },
                'flops': {                            # Computational requirements
                    'attention': float,               # FLOPS for attention calculations
                    'feedforward': float,             # FLOPS for feedforward calculations
                    'prefill_total': float,           # Total FLOPS for prefill phase
                    'per_token': float                # FLOPS per token during generation
                },
                'performance': {                      # Performance estimates
                    'tokens_per_second': float,       # Estimated tokens per second
                    'prefill_latency': float,         # Estimated prefill latency in seconds
                    'token_latency': float            # Estimated per-token latency in seconds
                },
                'parallelism': {                      # Parallelism configuration
                    'strategy': str,                  # Parallelism strategy (none, tensor, pipeline, etc.)
                    'tp_size': int,                   # Tensor parallelism size
                    'pp_size': int,                   # Pipeline parallelism size
                    'num_gpus': int,                  # Total number of GPUs
                    'effective_tflops': float         # Combined TFLOPS across all GPUs
                },
                'overheads_used': {                   # Overhead factors used in calculations
                    'weights': float,                 # Overhead for weights (typically 1.05)
                    'kv_cache': float,                # Overhead for KV cache (typically 1.05)
                    'activations': float,             # Overhead for activations (typically 1.10)
                    'system': float                   # System-wide overhead (typically 1.05)
                },
                'history': list                       # Calculation history from the calculator
            }
        """
        if not analysis:
            return {}
            
        # Extract model parameters from analysis
        model_name = data.get('model_name', 'custom')
        model_info = analysis.get('model_info', {})
        app.logger.info(f"[standardize] Received model_info: {json.dumps(model_info, indent=2)}") # Log model_info
        parameters_billions = model_info.get('parameters_b', 0)
        app.logger.info(f"[standardize] Extracted parameters_billions: {parameters_billions}") # Log extracted value
        
        # Setup default overheads
        default_overheads = {
            "weights": 1.05,
            "kv_cache": 1.05,
            "activations": 1.1,
            "system": 1.05
        }
        overheads = analysis.get('overheads_used', default_overheads)
        
        # Create standardized VRAM structure
        std_vram = {}
        raw_vram_data = analysis.get('vram', {})
        app.logger.info(f"Raw VRAM data received for standardization: {json.dumps(raw_vram_data, indent=2)}")

        # Check if the raw VRAM data seems valid (has the structure from calculate_total_vram)
        if isinstance(raw_vram_data, dict) and 'weights_base' in raw_vram_data and 'total' in raw_vram_data:
            # Directly use the keys from the structure returned by calculate_total_vram
            vram = raw_vram_data
            
            # Get base values
            std_vram['model_base'] = vram.get('weights_base', 0)
            std_vram['kv_cache_base'] = vram.get('kv_cache_base', 0)
            std_vram['activations_base'] = vram.get('activations_base', 0)
            std_vram['total_base'] = vram.get('total_base', 0)
            
            # Get with_overhead values
            std_vram['model_with_overhead'] = vram.get('weights_with_overhead', 0)
            std_vram['kv_cache_with_overhead'] = vram.get('kv_cache_with_overhead', 0)
            std_vram['activations_with_overhead'] = vram.get('activations_with_overhead', 0)
            std_vram['total_with_component_overhead'] = vram.get('component_subtotal', 0) # Use the subtotal before system overhead
            std_vram['total_system_wide'] = vram.get('total', 0) # Use the final total including system overhead

            # Extract per_gpu data if available, otherwise calculate for single GPU
            per_gpu_data = vram.get('per_gpu', {}) # Check for potential future 'per_gpu' dict from vram calculator
            num_gpus = analysis.get('parallelism', {}).get('num_gpus', 1) # Use the num_gpus from the standardized parallelism info

            # --- MODIFIED LOGIC FOR PER_GPU (Reflects base/overhead separation) --- 
            if isinstance(per_gpu_data, dict) and per_gpu_data and num_gpus > 1:
                 # Placeholder: If calculator starts providing per_gpu data, adapt key access here.
                 # Assumes per_gpu data will have similar base/overhead keys.
                 app.logger.warning("Received per_gpu data structure, but standardization logic needs verification/update based on actual keys.")
                 std_vram['model_per_gpu'] = per_gpu_data.get('weights_base', 0)
                 std_vram['kv_cache_per_gpu'] = per_gpu_data.get('kv_cache_base', 0)
                 std_vram['activations_per_gpu'] = per_gpu_data.get('activations_base', 0)
                 std_vram['model_per_gpu_with_overhead'] = per_gpu_data.get('weights_with_overhead', 0)
                 std_vram['kv_cache_per_gpu_with_overhead'] = per_gpu_data.get('kv_cache_with_overhead', 0)
                 std_vram['activations_per_gpu_with_overhead'] = per_gpu_data.get('activations_with_overhead', 0)
                 std_vram['total_base_per_gpu'] = per_gpu_data.get('total_base', 0)
                 std_vram['total_per_gpu_with_component_overhead'] = per_gpu_data.get('component_subtotal', 0)

            else:
                # Default to system-wide values if single GPU or no per_gpu data
                # Uses the base and overhead values calculated earlier
                std_vram['model_per_gpu'] = std_vram['model_base']
                std_vram['kv_cache_per_gpu'] = std_vram['kv_cache_base']
                std_vram['activations_per_gpu'] = std_vram['activations_base']
                std_vram['model_per_gpu_with_overhead'] = std_vram['model_with_overhead']
                std_vram['kv_cache_per_gpu_with_overhead'] = std_vram['kv_cache_with_overhead']
                std_vram['activations_per_gpu_with_overhead'] = std_vram['activations_with_overhead']
                std_vram['total_base_per_gpu'] = std_vram['total_base']
                std_vram['total_per_gpu_with_component_overhead'] = std_vram['total_with_component_overhead']
            # --- END MODIFIED PER_GPU LOGIC --- 

            app.logger.info(f"Successfully standardized VRAM data: {json.dumps(std_vram, indent=2)}")

        else:
            # Fallback: If VRAM data is missing or in an unexpected format, create zeros
            app.logger.warning(f"VRAM data missing 'weights_base' or 'total' key or not a dict: {raw_vram_data}")
            std_vram = {
                'model_base': 0, 'kv_cache_base': 0, 'activations_base': 0,
                'model_with_overhead': 0, 'kv_cache_with_overhead': 0, 'activations_with_overhead': 0,
                'total_base': 0, 'total_with_component_overhead': 0, 'total_system_wide': 0,
                'model_per_gpu': 0, 'kv_cache_per_gpu': 0, 'activations_per_gpu': 0,
                'model_per_gpu_with_overhead': 0, 'kv_cache_per_gpu_with_overhead': 0, 'activations_per_gpu_with_overhead': 0,
                'total_base_per_gpu': 0, 'total_per_gpu_with_component_overhead': 0
            }
            
        # Apply parallelism transformations if needed (This might be redundant now, but let's keep it for safety)
        # Note: The logic above tries to handle per_gpu data directly if available.
        # This block might adjust single-GPU figures if parallelism was requested *but* the vram_calc didn't provide per_gpu dict.
        if data.get('parallelism') and std_vram.get('total_system_wide', 0) > 0 : # Only apply if parallelism requested AND we have some VRAM numbers
            parallelism_data = data.get('parallelism', {})
            strategy = parallelism_data.get('strategy', 'none')
            tp_size = int(parallelism_data.get('tp_size', 1))
            pp_size = int(parallelism_data.get('pp_size', 1))
            num_gpus = int(parallelism_data.get('num_gpus', 1))
            
            # Check if per_gpu data was already handled, if not, apply simple division
            # This is a safety net in case vram_results['per_gpu'] wasn't populated as expected
            if num_gpus > 1 and std_vram.get('total_per_gpu_with_component_overhead') == std_vram.get('total_with_component_overhead'):
                app.logger.info("Applying post-hoc parallelism division to VRAM figures.") # Log if this safety net is triggered
                # KV cache and activations are typically replicated or handled differently,
                # focus on dividing model weights as a primary factor.
                # More complex logic might be needed in vram_calc itself for accurate distribution.
                
                # Recalculate model per_gpu based on strategy
                if strategy == 'tensor':
                    std_vram['model_per_gpu'] = std_vram.get('model_base', 0) / tp_size
                    std_vram['model_per_gpu_with_overhead'] = std_vram.get('model_with_overhead', 0) / tp_size
                elif strategy == 'pipeline':
                     std_vram['model_per_gpu'] = std_vram.get('model_base', 0) / pp_size
                     std_vram['model_per_gpu_with_overhead'] = std_vram.get('model_with_overhead', 0) / pp_size
                elif strategy == 'tensor_pipeline':
                    std_vram['model_per_gpu'] = std_vram.get('model_base', 0) / (tp_size * pp_size)
                    std_vram['model_per_gpu_with_overhead'] = std_vram.get('model_with_overhead', 0) / (tp_size * pp_size)
                
                # Simple recalculation of per_gpu totals (might not be perfectly accurate for complex cases)
                std_vram['total_base_per_gpu'] = (
                    std_vram.get('model_per_gpu', 0) + 
                    std_vram.get('kv_cache_per_gpu', 0) + # Assumes KV/Activations per_gpu were set correctly earlier or are system wide figures
                    std_vram.get('activations_per_gpu', 0)
                )
                std_vram['total_per_gpu_with_component_overhead'] = (
                    std_vram.get('model_per_gpu_with_overhead', 0) + 
                    std_vram.get('kv_cache_per_gpu_with_overhead', 0) +
                    std_vram.get('activations_per_gpu_with_overhead', 0)
                )

        # Standardize parallelism info
        parallelism = {
            'strategy': data.get('parallelism', {}).get('strategy', 'none'),
            'tp_size': int(data.get('parallelism', {}).get('tp_size', 1)),
            'pp_size': int(data.get('parallelism', {}).get('pp_size', 1)),
            'num_gpus': int(data.get('parallelism', {}).get('num_gpus', 1)),
            'effective_tflops': float(data.get('gpu', 312.0)) * int(data.get('parallelism', {}).get('num_gpus', 1))
        }
        
        # Standardize FLOPs data (ensure prefill_total exists)
        flops = analysis.get('flops', {})
        if 'prefill' in flops and 'prefill_total' not in flops:
            flops['prefill_total'] = flops['prefill']
        
        # Create final standardized response
        standardized = {
            'model_name': model_name,
            'parameters_billions': parameters_billions,
            'vram': std_vram,
            'flops': flops,
            'performance': analysis.get('performance', {}),
            'parallelism': parallelism,
            'overheads_used': overheads,
            'history': analysis.get('history', [])
        }
        
        app.logger.info(f"[standardize] Final standardized response: {json.dumps(standardized, indent=2)}") # Log final standardized dict
        
        return standardized

    def unifying_calculator(data):
        """
        Single point of entry for all calculator operations.
        
        Args:
            data: Dictionary of calculation parameters
            
        Returns:
            Standardized calculator response
        """
        # app.logger.info(f"Calculator input data: {data}") # Redundant log: Input logged in /api/calculate
        
        # Extract core parameters
        model_name = data.get('model_name', 'custom')
        hidden_dim = int(data.get('hidden_dim', 0))
        ff_dim = int(data.get('ff_dim', 0))
        num_layers = int(data.get('num_layers', 0))
        vocab_size = int(data.get('vocab_size', 0))
        seq_length = max(1, int(data.get('seq_length', 2048)))  # Ensure minimum of 1
        batch_size = int(data.get('batch_size', 1))
        precision = data.get('precision', 'fp16')
        gpu_id = data.get('gpu', '')
        efficiency_factor = float(data.get('efficiency_factor', 0.3))
        
        try:
            # Choose correct calculator method based on GPU ID
            if gpu_id and isinstance(gpu_id, str) and not gpu_id.replace('.', '').isdigit():
                # Try with a named GPU first
                try:
                    gpu_config = get_gpu_config(gpu_id)
                    if gpu_config:
                        analysis = calculator.analyze_model_on_gpu(
                            model_name=model_name,
                            gpu_name=gpu_id,
                            sequence_length=seq_length,
                            batch_size=batch_size,
                            precision=precision,
                            efficiency_factor=efficiency_factor
                        )
                        return standardize_calculator_response(analysis, data)
                    
                except ValueError as e:
                    # If GPU is not recognized, fall back to using TFLOPS value
                    app.logger.warning(f"GPU '{gpu_id}' not recognized: {str(e)}. Falling back to TFLOPS.")
            
            # Fall back to model analysis with TFLOPS value
            gpu_tflops = 312.0  # Default to A100
            
            # Try to convert GPU ID to TFLOPS if it's a number
            if gpu_id:
                try:
                    maybe_tflops = float(gpu_id)
                    if maybe_tflops > 0:
                        gpu_tflops = maybe_tflops
                except (ValueError, TypeError):
                    pass  # Use default TFLOPS value if conversion fails
            
            # Analyze with explicit TFLOPS value
            analysis = calculator.analyze_model_by_name(
                model_name=model_name,
                sequence_length=seq_length,
                batch_size=batch_size,
                precision=precision,
                gpu_tflops=gpu_tflops,
                efficiency_factor=efficiency_factor
            )
            
            return standardize_calculator_response(analysis, data)
            
        except Exception as e:
            app.logger.error(f"Calculator error: {str(e)}")
            # Return error information that can be processed by the frontend
            # Return raw analysis dictionary on success
            return analysis

    def get_gpu_configs():
        """
        Get GPU configurations from gpus module.
        
        Returns:
            List of dictionaries containing GPU configurations with:
            - name: GPU model name
            - tflops: Performance in teraFLOPs
            - vram: VRAM capacity in GB
            - family: GPU family/architecture
            - supported_precisions: List of supported precision formats
        """
        try:
            # Get all available GPUs from the module
            gpu_list = list_all_gpus()
            result = []
            
            for gpu_id in gpu_list:
                gpu_config = get_gpu_config(gpu_id)
                if gpu_config:
                    # Convert to format expected by the UI
                    result.append({
                        "name": gpu_config["name"],
                        "tflops": gpu_config.get("fp16_tflops", gpu_config.get("bf16_tflops", 0)),
                        "vram": gpu_config.get("vram_gb", 0),
                        "family": gpu_config.get("family", ""),
                        "gen": gpu_config.get("gen", ""),
                        "id": gpu_id,
                        "supported_precisions": gpu_config.get("supported_precisions", ["fp32", "fp16", "bf16"])
                    })
            
            # Sort by family and VRAM
            result.sort(key=lambda x: (x["family"], -x["vram"]))
            return result
        except Exception as e:
            app.logger.error(f"Error loading GPU configurations: {e}")
            # Return a minimal set of GPUs if the module fails
            return [
                {"name": "A100 (80GB)", "tflops": 312.0, "vram": 80, "family": "NVIDIA", "gen": "Ampere", "supported_precisions": ["fp32", "fp16", "bf16"]},
                {"name": "H100", "tflops": 756.0, "vram": 80, "family": "NVIDIA", "gen": "Hopper", "supported_precisions": ["fp32", "fp16", "bf16", "fp8"]}
            ]

    def get_available_models():
        """
        Get model configurations from models module.
        
        Returns:
            List of dictionaries containing model configurations with:
            - name: Model name
            - hidden_dim: Hidden dimension size
            - ff_dim: Feedforward dimension size
            - num_layers: Number of transformer layers
            - vocab_size: Vocabulary size
            - seq_len: Maximum sequence length
            - family: Model family name
            - parameter_count: Number of parameters in billions
        """
        try:
            # Get all available models from the module
            model_list = list_all_models()
            result = []
            
            for model_id in model_list:
                model_config = get_model_config(model_id)
                if model_config:
                    # Convert to format expected by the UI
                    result.append({
                        "name": model_id,
                        "hidden_dim": model_config.get("hidden_dimensions", 0),
                        "ff_dim": model_config.get("feedforward_dimensions", 0),
                        "num_layers": model_config.get("num_layers", 0),
                        "vocab_size": model_config.get("vocab_size", 0),
                        "seq_len": model_config.get("max_sequence_length", 0),
                        "family": model_config.get("family", "Unknown"),
                        "parameter_count": model_config.get("parameters_billions", 0)
                    })
            
            # Sort by family and parameter count
            result.sort(key=lambda x: (x["family"], x["parameter_count"]))
            return result
        except Exception as e:
            app.logger.error(f"Error loading model configurations: {e}")
            # Return a minimal set of models if the module fails
            return [
                {"name": "llama2-7b", "hidden_dim": 4096, "ff_dim": 11008, "num_layers": 32, "vocab_size": 32000, "seq_len": 4096, "family": "Llama", "parameter_count": 7.0},
                {"name": "llama2-13b", "hidden_dim": 5120, "ff_dim": 13824, "num_layers": 40, "vocab_size": 32000, "seq_len": 4096, "family": "Llama", "parameter_count": 13.0},
                {"name": "llama2-70b", "hidden_dim": 8192, "ff_dim": 28672, "num_layers": 80, "vocab_size": 32000, "seq_len": 4096, "family": "Llama", "parameter_count": 70.0},
            ]

    def get_model_families_list():
        """
        Get list of model families.
        
        Returns:
            List of model family names
        """
        try:
            return get_model_families()
        except Exception as e:
            app.logger.error(f"Error getting model families: {e}")
            return ["Llama", "Mistral", "Phi"]

    @app.route('/')
    def index():
        """Render the main page of the LLM calculator web interface."""
        app.logger.info("Main page accessed")
        return render_template('index.html', 
                               gpu_configs=get_gpu_configs(),
                               models=get_available_models(),
                               model_families=get_model_families_list())

    @app.route('/model_calculation', methods=['POST'])
    def model_calculation():
        """API endpoint for model calculation"""
        content_type = request.headers.get('Content-Type')
        if content_type != 'application/json':
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        try:
            data = request.json
            analysis = unifying_calculator(data)
            
            # Use standardized response format
            standardized_response = standardize_calculator_response(analysis, data)
            
            return jsonify(standardized_response), 200
        except Exception as e:
            app.logger.error(f"Error in model_calculation: {str(e)}")
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    @app.route('/api/available_models', methods=['GET'])
    def available_models():
        """API endpoint to get available predefined models."""
        return jsonify(get_available_models())

    @app.route('/api/model_scaling_analysis', methods=['POST'])
    def model_scaling_analysis():
        """API endpoint to analyze scaling a model across different configurations."""
        try:
            if not request.is_json:
                return jsonify({"error": "Request must be JSON"}), 400
                
            data = request.json
            app.logger.info(f"Model scaling analysis request: {data}")
            
            # Extract request parameters - handle both direct parameters and nested objects
            model_config = data.get('model_config', {})
            gpu_config = data.get('gpu_config', {})
            
            # Get parameters from either nested objects or direct properties
            hidden_dim = int(model_config.get('hidden_dim', data.get('base_hidden_dim', 0)))
            ff_dim = int(model_config.get('ff_dim', data.get('base_ff_dim', 0)))
            num_layers = int(model_config.get('num_layers', data.get('base_num_layers', 0)))
            vocab_size = int(model_config.get('vocab_size', data.get('base_vocab_size', 0)))
            seq_length = int(model_config.get('seq_len', data.get('sequence_length', data.get('base_seq_length', 2048))))
            
            # GPU parameters
            gpu_vram_gb = float(gpu_config.get('vram', data.get('gpu_vram_gb', 80.0)))
            batch_size = int(data.get('batch_size', 1))
            precision = data.get('precision', 'fp16')
            
            app.logger.info(f"Parsed model parameters: hidden_dim={hidden_dim}, ff_dim={ff_dim}, num_layers={num_layers}, vocab_size={vocab_size}, seq_length={seq_length}")
            app.logger.info(f"Parsed GPU parameters: vram={gpu_vram_gb}, batch_size={batch_size}, precision={precision}")
            
            # Use the determine_model_scaling method from calculator
            scaling_result = calculator.determine_model_scaling(
                gpu_vram_gb=gpu_vram_gb,
                batch_size=batch_size,
                sequence_length=seq_length,
                hidden_dimensions=hidden_dim,
                feedforward_dimensions=ff_dim,
                num_layers=num_layers,
                vocab_size=vocab_size,
                precision=precision
            )
            
            # Add safe defaults for any missing fields
            if scaling_result is None:
                scaling_result = {}
            
            # Ensure all expected fields exist
            scaling_result.setdefault('total_vram_required_gb', None)
            scaling_result.setdefault('gpu_vram_capacity_gb', gpu_vram_gb)
            scaling_result.setdefault('fits_on_single_gpu', False)
            scaling_result.setdefault('num_gpus_required', None)
            scaling_result.setdefault('recommended_strategy', None)
            
            if 'scaling_details' not in scaling_result:
                scaling_result['scaling_details'] = {'note': 'No detailed scaling information available'}
            
            # Log the analysis
            log_dir = "logs/model_scaling"
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"model_scaling_{timestamp}.json"
            
            log_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "input_parameters": data,
                "results": scaling_result
            }
            
            with open(os.path.join(log_dir, log_filename), 'w') as f:
                json.dump(log_data, f, indent=2)
            
            # Return the scaling analysis
            return jsonify(scaling_result)
                
        except Exception as e:
            app.logger.error(f"Error in model scaling analysis endpoint: {str(e)}")
            return jsonify({
                "error": str(e),
                "total_vram_required_gb": None,
                "gpu_vram_capacity_gb": None,
                "fits_on_single_gpu": False,
                "num_gpus_required": None,
                "recommended_strategy": None,
                "scaling_details": {"note": f"Analysis failed: {str(e)}"}
            }), 200  # Return 200 with error details in the response body

    @app.route('/api/gpu_configs', methods=['GET'])
    def gpu_configs():
        """API endpoint to get available GPU configurations."""
        return jsonify(get_gpu_configs())

    @app.route('/api/model_config/<model_name>', methods=['GET'])
    def model_config(model_name):
        """API endpoint to get configuration for a specific model."""
        try:
            # First, check the predefined models
            all_models = get_available_models()
            for model in all_models:
                if model['name'].lower() == model_name.lower():
                    return jsonify(model)
            
            # If not found in predefined models, try to get from model_config
            model_config_data = get_model_config(model_name)
            if model_config_data:
                # Convert to the expected format
                return jsonify({
                    "name": model_name,
                    "hidden_dim": model_config_data.get("hidden_dimensions", 0),
                    "ff_dim": model_config_data.get("feedforward_dimensions", 0),
                    "num_layers": model_config_data.get("num_layers", 0),
                    "vocab_size": model_config_data.get("vocab_size", 0),
                    "seq_len": model_config_data.get("max_sequence_length", 0),
                    "family": model_config_data.get("family", "Unknown"),
                    "parameter_count": model_config_data.get("parameters_billions", 0)
                })
            
            # Return not found if the model doesn't exist
            return jsonify({"error": f"Model not found: {model_name}"}), 404
                
        except Exception as e:
            app.logger.error(f"Error fetching model config for {model_name}: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @app.route('/visualize')
    def visualize_results():
        """Render the results visualization page."""
        app.logger.info("Visualization page accessed")
        return render_template('visualize.html')

    # Log file parsing functions for visualization
    def extract_data_from_log_line(line, log_type):
        """Extract relevant data from a log line based on the log type."""
        if log_type == "calculations":
            # For calculation logs (JSON format)
            try:
                data = json.loads(line)
                # Extract timestamp and key metrics
                timestamp = data.get("timestamp", "")
                if "results" in data:
                    results = data["results"]
                    if "summary" in results:
                        summary = results["summary"]
                        return {
                            "timestamp": timestamp,
                            "model_size": summary.get("model_size_billions", 0),
                            "gpus_needed": summary.get("gpus_needed", 0),
                            "training_days": summary.get("training_days", 0),
                            "training_cost": summary.get("training_cost_usd", 0)
                        }
            except json.JSONDecodeError:
                pass
        return None

    def parse_log_file(log_path, log_type):
        """Parse a log file and extract relevant data for visualization."""
        if not os.path.exists(log_path):
            return []
        
        data_points = []
        try:
            with open(log_path, 'r') as f:
                for line in f:
                    data_point = extract_data_from_log_line(line.strip(), log_type)
                    if data_point:
                        data_points.append(data_point)
        except Exception as e:
            app.logger.error(f"Error parsing log file {log_path}: {str(e)}")
        
        return data_points

    @app.route('/api/calculate', methods=['POST'])
    def calculate():
        """API endpoint to calculate LLM requirements based on submitted parameters."""
        try:
            if not request.is_json:
                return jsonify({"error": "Request must be JSON"}), 400
            
            data = request.json
            app.logger.info(f"Calculation request: {json.dumps(data, indent=2)}")
            
            # Extract request parameters
            calculation_type = data.get('calculation_type', 'model')
            
            if calculation_type == 'model':
                # Use the unified calculator interface
                analysis = unifying_calculator(data) # Returns standardized analysis or error dict
                
                # Check if unifying_calculator returned an error dictionary
                if isinstance(analysis, dict) and analysis.get("error"):
                     app.logger.error(f"Calculation failed: {analysis['error']}")
                     # Return the error structure directly
                     return jsonify(analysis), 500 # Send 500 for calculator errors

                # Log calculation results (analysis is already standardized)
                log_dir = "logs/calculations"
                os.makedirs(log_dir, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = data.get('model_name', 'custom')
                log_filename = f"web_calc_{model_name}_{timestamp}.json"
                
                # Use the already standardized response from unifying_calculator
                standardized_response = analysis # No need to call standardize_calculator_response again
                
                log_data = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "input_parameters": data,
                    "results": standardized_response # Log the standardized results
                }
                
                with open(os.path.join(log_dir, log_filename), 'w') as f:
                    json.dump(log_data, f, indent=2)
                
                # Return the standardized analysis results
                # app.logger.info(f"[/api/calculate] Response data being sent: {standardized_response}") # Redundant log: Response logged during standardization
                return jsonify(standardized_response)
                
            else:
                return jsonify({"error": f"Unsupported calculation type: {calculation_type}"}), 400
                
        except Exception as e:
            app.logger.error(f"Error in calculate endpoint: {str(e)}")
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    return app 