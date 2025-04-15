#!/usr/bin/env python3
import sys
import os
import logging
import json
import re
import datetime
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

    @app.route('/api/calculate', methods=['POST'])
    def calculate():
        """API endpoint to calculate LLM requirements based on submitted parameters."""
        try:
            if not request.is_json:
                return jsonify({"error": "Request must be JSON"}), 400
                
            data = request.json
            app.logger.info(f"Calculation request: {data}")
            
            # Extract request parameters
            calculation_type = data.get('calculation_type', 'model')
            
            if calculation_type == 'model':
                # For model calculations, extract all the parameters
                hidden_dim = int(data.get('hidden_dim', 0))
                ff_dim = int(data.get('ff_dim', 0))
                num_layers = int(data.get('num_layers', 0))
                vocab_size = int(data.get('vocab_size', 0))
                seq_length = int(data.get('seq_length', 2048))
                batch_size = int(data.get('batch_size', 1))
                
                # GPU parameters
                gpu_id = data.get('gpu', '')
                precision = data.get('precision', 'fp16')
                efficiency_factor = float(data.get('efficiency_factor', 0.3))
                
                # If a GPU was selected, analyze model on that GPU
                if gpu_id:
                    analysis = calculator.analyze_model_on_gpu(
                        model_name=data.get('model_name', 'custom'),  # Use 'custom' for custom models
                        gpu_name=gpu_id,
                        sequence_length=seq_length,
                        batch_size=batch_size,
                        precision=precision,
                        efficiency_factor=efficiency_factor
                    )
                else:
                    # Create model parameters manually
                    model_params = {
                        'hidden_dimensions': hidden_dim,
                        'feedforward_dimensions': ff_dim,
                        'num_layers': num_layers,
                        'vocab_size': vocab_size,
                        'max_sequence_length': seq_length,
                    }
                    
                    # Get selected GPU info or use default
                    gpu_tflops = float(data.get('gpu_tflops', 312.0))  # Default to A100
                    
                    # Analyze model with custom params
                    analysis = calculator.analyze_model_by_name(
                        model_name=data.get('model_name', 'custom'),
                        sequence_length=seq_length,
                        batch_size=batch_size,
                        precision=precision,
                        gpu_tflops=gpu_tflops,
                        efficiency_factor=efficiency_factor
                    )
                
                # Transform VRAM data for frontend
                # Ensure we have properly structured data for both per_gpu and system_wide calculations
                # Frontend should not need to perform any calculations
                if "vram" in analysis:
                    # Calculate pre-computed totals before applying system overhead
                    if isinstance(analysis["vram"].get("model"), dict):
                        # Structure from dictionary return type
                        analysis["vram"]["model_base"] = analysis["vram"]["model"].get("weights_base", 0)
                        analysis["vram"]["kv_cache_base"] = analysis["vram"]["model"].get("kv_cache_base", 0)
                        analysis["vram"]["activations_base"] = analysis["vram"]["model"].get("activations_base", 0)
                        
                        analysis["vram"]["model_with_overhead"] = analysis["vram"]["model"].get("weights_with_overhead", 0)
                        analysis["vram"]["kv_cache_with_overhead"] = analysis["vram"]["model"].get("kv_cache_with_overhead", 0)
                        analysis["vram"]["activations_with_overhead"] = analysis["vram"]["model"].get("activations_with_overhead", 0)
                        
                        # Total before system overhead
                        analysis["vram"]["total_base"] = (
                            analysis["vram"]["model_base"] + 
                            analysis["vram"]["kv_cache_base"] + 
                            analysis["vram"]["activations_base"]
                        )
                        
                        analysis["vram"]["total_with_component_overhead"] = analysis["vram"]["model"].get("component_subtotal", 0)
                        analysis["vram"]["total_system_wide"] = analysis["vram"]["model"].get("total", 0)
                    
                    # Ensure we have per_gpu values properly structured too 
                    if "model_per_gpu" in analysis["vram"]:
                        analysis["vram"]["total_base_per_gpu"] = (
                            analysis["vram"]["model_per_gpu"] + 
                            analysis["vram"]["kv_cache_per_gpu"] + 
                            analysis["vram"]["activations_per_gpu"]
                        )
                        
                        # Add component overhead values if not present
                        if "model_per_gpu_with_overhead" not in analysis["vram"] and "overheads_used" in analysis:
                            analysis["vram"]["model_per_gpu_with_overhead"] = (
                                analysis["vram"]["model_per_gpu"] * analysis["overheads_used"].get("weights", 1.0)
                            )
                            analysis["vram"]["kv_cache_per_gpu_with_overhead"] = (
                                analysis["vram"]["kv_cache_per_gpu"] * analysis["overheads_used"].get("kv_cache", 1.0)
                            )
                            analysis["vram"]["activations_per_gpu_with_overhead"] = (
                                analysis["vram"]["activations_per_gpu"] * analysis["overheads_used"].get("activations", 1.0)
                            )
                            
                            # Calculate component subtotal
                            analysis["vram"]["total_per_gpu_with_component_overhead"] = (
                                analysis["vram"]["model_per_gpu_with_overhead"] + 
                                analysis["vram"]["kv_cache_per_gpu_with_overhead"] +
                                analysis["vram"]["activations_per_gpu_with_overhead"]
                            )
                
                # Add calculation history
                history = calculator.get_history()
                
                # Log calculation results
                log_dir = "logs/calculations"
                os.makedirs(log_dir, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = data.get('model_name', 'custom')
                log_filename = f"web_calc_{model_name}_{timestamp}.json"
                
                log_data = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "input_parameters": data,
                    "results": analysis
                }
                
                with open(os.path.join(log_dir, log_filename), 'w') as f:
                    json.dump(log_data, f, indent=2)
                
                # Return the analysis results and calculation history
                return jsonify({
                    "model_name": data.get('model_name', 'custom'),
                    "parameters_billions": analysis.get("parameters_billions", 0),
                    "vram": analysis["vram"],
                    "flops": analysis["flops"],
                    "performance": analysis["performance"],
                    "parallelism": analysis["parallelism"],
                    "overheads_used": analysis["overheads_used"],
                    "history": history
                })
                
            else:
                return jsonify({"error": f"Unsupported calculation type: {calculation_type}"}), 400
                
        except Exception as e:
            app.logger.error(f"Error in calculate endpoint: {str(e)}")
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
            
            # Extract request parameters
            base_hidden_dim = int(data.get('base_hidden_dim', 0))
            base_ff_dim = int(data.get('base_ff_dim', 0))
            base_num_layers = int(data.get('base_num_layers', 0))
            base_vocab_size = int(data.get('base_vocab_size', 0))
            base_seq_length = int(data.get('base_seq_length', 2048))
            gpu_vram_gb = float(data.get('gpu_vram_gb', 80.0))
            batch_size = int(data.get('batch_size', 1))
            precision = data.get('precision', 'fp16')
            
            # Use the determine_model_scaling method from calculator
            scaling_result = calculator.determine_model_scaling(
                gpu_vram_gb=gpu_vram_gb,
                batch_size=batch_size,
                sequence_length=base_seq_length,
                hidden_dimensions=base_hidden_dim,
                feedforward_dimensions=base_ff_dim,
                num_layers=base_num_layers,
                vocab_size=base_vocab_size,
                precision=precision
            )
            
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
            return jsonify({"error": str(e)}), 500

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

    return app 