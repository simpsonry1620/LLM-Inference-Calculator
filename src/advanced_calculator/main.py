from typing import List, Union, Optional, Any, Literal, Dict

from src.advanced_calculator.modules.flops import FLOPsCalculator
from src.advanced_calculator.modules.vram import VRAMCalculator
from src.advanced_calculator.modules.throughput import ThroughputCalculator
from src.advanced_calculator.modules.latency import LatencyCalculator
from src.advanced_calculator.modules.utils import HistoryManager
from src.advanced_calculator.modules.models import (
    get_model_config, list_all_models, get_model_families, get_models_by_family, ModelConfig
)
from src.advanced_calculator.modules.gpus import (
    get_gpu_config, list_all_gpus, get_gpu_families, get_gpu_generations,
    get_gpus_by_family, get_gpus_by_generation, get_gpus_by_min_vram,
    get_gpus_supporting_precision, get_recommended_gpu_for_model, GPUConfig
)

class AdvancedCalculator:
    """
    Advanced calculator for estimating computational requirements of large language models.
    
    This calculator provides methods to estimate:
    - FLOPs (Floating Point Operations) for various components of transformer models
    - VRAM requirements for model weights and inference
    - Throughput in tokens per second
    - Latency for prefill and token generation
    
    These calculations can be useful for planning infrastructure requirements,
    estimating training and inference costs, and understanding scaling properties
    of large language models.
    """
    def __init__(self) -> None:
        """Initialize the advanced calculator with its component calculators"""
        self._history = HistoryManager()
        
        # Initialize component calculators with history callback
        self._flops = FLOPsCalculator(history_callback=self._history.add_entry)
        self._vram = VRAMCalculator(history_callback=self._history.add_entry)
        self._throughput = ThroughputCalculator(history_callback=self._history.add_entry)
        self._latency = LatencyCalculator(history_callback=self._history.add_entry)
    
    # Model selection methods
    def get_available_models(self) -> List[str]:
        """
        Get a list of available predefined model names.
        
        Returns:
            List of available model names
        """
        return list_all_models()
    
    def get_model_families(self) -> List[str]:
        """
        Get a list of available model families.
        
        Returns:
            List of model family names
        """
        return get_model_families()
    
    def get_models_by_family(self, family: str) -> List[Dict[str, Any]]:
        """
        Get all models belonging to a specific family.
        
        Args:
            family: Name of the model family
            
        Returns:
            List of model configurations in the specified family
        """
        return get_models_by_family(family)
    
    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific model by name.
        
        Args:
            model_name: Name of the model to retrieve
            
        Returns:
            Model configuration dictionary or None if not found
        """
        return get_model_config(model_name)
    
    # GPU selection methods
    def get_available_gpus(self) -> List[str]:
        """
        Get a list of available predefined GPU names.
        
        Returns:
            List of available GPU names
        """
        return list_all_gpus()
    
    def get_gpu_families(self) -> List[str]:
        """
        Get a list of available GPU families.
        
        Returns:
            List of GPU family names
        """
        return get_gpu_families()
    
    def get_gpu_generations(self) -> List[str]:
        """
        Get a list of available GPU generations.
        
        Returns:
            List of GPU generation names
        """
        return get_gpu_generations()
    
    def get_gpus_by_family(self, family: str) -> List[Dict[str, Any]]:
        """
        Get all GPUs belonging to a specific family.
        
        Args:
            family: Name of the GPU family
            
        Returns:
            List of GPU configurations in the specified family
        """
        return get_gpus_by_family(family)
    
    def get_gpus_by_generation(self, generation: str) -> List[Dict[str, Any]]:
        """
        Get all GPUs belonging to a specific generation.
        
        Args:
            generation: Name of the GPU generation
            
        Returns:
            List of GPU configurations in the specified generation
        """
        return get_gpus_by_generation(generation)
    
    def get_gpu_config(self, gpu_name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific GPU by name.
        
        Args:
            gpu_name: Name of the GPU to retrieve
            
        Returns:
            GPU configuration dictionary or None if not found
        """
        return get_gpu_config(gpu_name)
    
    def get_gpus_by_min_vram(self, min_vram_gb: float) -> List[Dict[str, Any]]:
        """
        Get all GPUs with at least the specified amount of VRAM.
        
        Args:
            min_vram_gb: Minimum VRAM in gigabytes
            
        Returns:
            List of GPU configurations with sufficient VRAM
        """
        return get_gpus_by_min_vram(min_vram_gb)
    
    def get_gpus_supporting_precision(self, precision: str) -> List[Dict[str, Any]]:
        """
        Get all GPUs that support the specified precision.
        
        Args:
            precision: Precision to check for (e.g., "fp16", "bf16", "fp8")
            
        Returns:
            List of GPU configurations supporting the precision
        """
        return get_gpus_supporting_precision(precision)
    
    def get_recommended_gpus_for_model(self, 
                                      model_name_or_vram: Union[str, float], 
                                      min_vram_headroom_gb: float = 2.0) -> List[Dict[str, Any]]:
        """
        Get recommended GPUs for a model.
        
        Args:
            model_name_or_vram: Either a predefined model name or VRAM requirements in GB
            min_vram_headroom_gb: Minimum extra VRAM headroom to recommend
            
        Returns:
            List of recommended GPU configurations sorted by efficiency
            
        Raises:
            ValueError: If the model name is not recognized
        """
        # Determine VRAM requirements
        if isinstance(model_name_or_vram, str):
            # Get model by name
            model_config = get_model_config(model_name_or_vram)
            if not model_config:
                raise ValueError(f"Model '{model_name_or_vram}' not recognized")
            
            # Get model parameters
            hidden_dim = model_config["hidden_dimensions"]
            ff_dim = model_config["feedforward_dimensions"]
            num_layers = model_config["num_layers"]
            vocab_size = model_config["vocab_size"]
            
            # Calculate model VRAM
            model_vram_gb = self.calculate_model_vram(
                hidden_dimensions=hidden_dim,
                feedforward_dimensions=ff_dim,
                num_layers=num_layers,
                vocab_size=vocab_size
            )
            
            min_vram_gb = model_vram_gb + min_vram_headroom_gb
        else:
            # Use directly specified VRAM requirements
            min_vram_gb = model_name_or_vram + min_vram_headroom_gb
            
        # Get GPUs with sufficient VRAM
        return get_gpus_by_min_vram(min_vram_gb)
        
    def analyze_model_on_gpu(self, 
                           model_name: str, 
                           gpu_name: str,
                           input_sequence_length: Optional[int] = None,
                           output_sequence_length: int = 128,
                           batch_size: int = 1,
                           precision: Literal["fp16", "fp32", "bf16", "int8", "int4"] = "fp16",
                           efficiency_factor: float = 0.3) -> Dict[str, Any]:
        """
        Analyze a model's performance on a specific GPU.
        
        Args:
            model_name: Name of the predefined model
            gpu_name: Name of the GPU to analyze
            input_sequence_length: Input sequence length (uses model default if not specified)
            output_sequence_length: Desired output sequence length in tokens
            batch_size: Number of sequences processed in parallel
            precision: Data precision for inference
            efficiency_factor: Computation efficiency factor (0-1)
            
        Returns:
            Dictionary with detailed analysis results, including keys like:
            - "model_info": Dictionary with model details (name, family, parameters_b, dimensions, etc.)
            - "gpu_info": Dictionary with GPU details (name, family, vram_gb, tflops, etc.)
            - "analysis_parameters": Dictionary with input parameters used for analysis
            - "flops": Dictionary with FLOPs breakdown (attention, feedforward, prefill, per_token)
            - "vram": Dictionary with VRAM breakdown (weights, kv_cache, activations, total, etc.)
            - "performance": Dictionary with performance estimates (tokens_per_second, latency, etc.)
            - "compatibility": Dictionary with GPU compatibility info (fits_on_gpu, headroom, etc.)
            - "overheads_used": Dictionary with overhead factors applied
            
        Raises:
            ValueError: If the model or GPU name is not recognized
        """
        # Get model configuration
        model_config = get_model_config(model_name)
        if not model_config:
            raise ValueError(f"Model '{model_name}' not recognized")
            
        # Get GPU configuration
        gpu_config = get_gpu_config(gpu_name)
        if not gpu_config:
            raise ValueError(f"GPU '{gpu_name}' not recognized")
            
        # Check if precision is supported by the GPU
        if precision not in gpu_config.get("supported_precisions", ["fp32"]):
            raise ValueError(f"Precision '{precision}' not supported by {gpu_name}")
            
        # Get GPU interconnect bandwidth
        interconnect_bandwidth = gpu_config.get("interconnect_bandwidth_gb_per_sec", 0.0)
        if interconnect_bandwidth == 0.0:
            # Add a warning or default if bandwidth info is missing? For now, raise error.
             raise ValueError(f"Interconnect bandwidth information missing for {gpu_name}")

        # Extract model parameters
        hidden_dim = model_config["hidden_dimensions"]
        ff_dim = model_config["feedforward_dimensions"]
        num_layers = model_config["num_layers"]
        vocab_size = model_config["vocab_size"]
        
        # Use model-specific sequence length if not specified
        current_input_length = input_sequence_length
        if current_input_length is None:
            current_input_length = model_config.get("max_sequence_length", 2048)
        
        # Validate sequence lengths are positive
        if current_input_length <= 0:
            raise ValueError(f"Input sequence length must be positive, got {current_input_length}")
        if output_sequence_length <= 0:
            raise ValueError(f"Output sequence length must be positive, got {output_sequence_length}")
            
        # Get GPU performance based on precision
        if precision in ["fp16", "bf16"]:
            gpu_tflops = gpu_config.get("fp16_tflops", gpu_config.get("bf16_tflops", 0))
        elif precision == "fp32":
            gpu_tflops = gpu_config.get("fp32_tflops", 0)
        elif precision == "int8":
            gpu_tflops = gpu_config.get("int8_tflops", gpu_config.get("fp16_tflops", 0) * 2)  # Estimate if not available
        elif precision == "int4":
            gpu_tflops = gpu_config.get("int4_tflops", gpu_config.get("fp16_tflops", 0) * 4)  # Estimate if not available
        else:
            gpu_tflops = 0
            
        if gpu_tflops == 0:
            raise ValueError(f"Performance data for {gpu_name} with {precision} precision not available")
            
        # Calculate FLOPs (Prefill depends on input length)
        flops_attention = self.calculate_flops_attention(batch_size, current_input_length, hidden_dim)
        flops_feedforward = self.calculate_flops_feedforward(batch_size, current_input_length, hidden_dim, ff_dim)
        flops_prefill = self.calculate_flops_prefill(batch_size, current_input_length, hidden_dim, ff_dim, num_layers)
        flops_per_token = self.calculate_flops_per_token(batch_size, hidden_dim, ff_dim, num_layers)
        
        # Default overhead factors for VRAM calculations
        weights_overhead = 1.05
        kv_cache_overhead = 1.05
        activations_overhead = 1.1
        system_overhead = 1.05
        
        # Calculate VRAM requirements (Pass both lengths now)
        vram_dict = self.calculate_total_vram(
            batch_size=batch_size,
            input_sequence_length=current_input_length,
            hidden_dimensions=hidden_dim,
            feedforward_dimensions=ff_dim,
            num_layers=num_layers,
            vocab_size=vocab_size,
            output_sequence_length=output_sequence_length,
            precision=precision,
            weights_overhead=weights_overhead,
            kv_cache_overhead=kv_cache_overhead,
            activations_overhead=activations_overhead,
            system_overhead=system_overhead
        )
        
        # Calculate performance metrics (Latency depends on input/output lengths)
        tokens_per_second = self.estimate_inference_throughput(flops_per_token, gpu_tflops, efficiency_factor)
        prefill_latency = self.calculate_prefill_latency(flops_prefill, gpu_tflops, efficiency_factor)
        token_latency = self.calculate_token_latency(flops_per_token, gpu_tflops, efficiency_factor)
        time_for_1000_tokens = token_latency * 1000
        
        # Calculate TTFT and Total Request Time
        time_to_first_token = prefill_latency
        total_request_time = prefill_latency + (output_sequence_length * token_latency)
        
        # Check if model fits on GPU
        total_vram_required = vram_dict["total"]
        gpu_vram = gpu_config["vram_gb"]
        fits_on_gpu = total_vram_required <= gpu_vram
        
        # Calculate throughput on different GPUs
        throughput_by_gpu = self.calculate_throughput_across_gpus(flops_per_token, efficiency_factor)
        
        # Create result dictionary
        return {
            "model_info": {
                "name": model_name,
                "family": model_config.get("family", "Unknown"),
                "parameters_b": model_config.get("parameter_count", 0),
                "hidden_dimensions": hidden_dim,
                "feedforward_dimensions": ff_dim,
                "num_layers": num_layers,
                "vocab_size": vocab_size,
                "description": model_config.get("description", "")
            },
            "gpu_info": {
                "name": gpu_name,
                "family": gpu_config.get("family", "Unknown"),
                "vram_gb": gpu_vram,
                "tflops": gpu_tflops,
                "interconnect_bandwidth_gb_per_sec": interconnect_bandwidth,
                "supported_precisions": gpu_config.get("supported_precisions", [])
            },
            "analysis_parameters": {
                "input_sequence_length": current_input_length,
                "output_sequence_length": output_sequence_length,
                "batch_size": batch_size,
                "precision": precision,
                "efficiency_factor": efficiency_factor
            },
            "flops": {
                "attention": flops_attention,
                "feedforward": flops_feedforward,
                "prefill_total": flops_prefill,
                "per_token": flops_per_token
            },
            "vram": vram_dict,
            "performance": {
                "tokens_per_second": tokens_per_second,
                "prefill_latency": prefill_latency,
                "token_latency": token_latency,
                "time_to_first_token": time_to_first_token,
                "total_request_time": total_request_time,
                "time_for_1000_tokens": time_for_1000_tokens
            },
            "compatibility": {
                "fits_on_gpu": fits_on_gpu,
                "vram_utilization_pct": (total_vram_required / gpu_vram) * 100 if gpu_vram > 0 else float('inf'),
                "headroom_gb": gpu_vram - total_vram_required if fits_on_gpu else 0,
                "minimum_required_vram_gb": total_vram_required
            },
            "overheads_used": {
                "weights": weights_overhead,
                "kv_cache": kv_cache_overhead,
                "activations": activations_overhead,
                "system": system_overhead
            }
        }
    
    def analyze_model_by_name(self, 
                             model_name: str, 
                             input_sequence_length: Optional[int] = None,
                             output_sequence_length: int = 128,
                             batch_size: int = 1,
                             precision: Literal["fp16", "fp32", "bf16"] = "fp16",
                             gpu_tflops: float = 312.0,
                             efficiency_factor: float = 0.3) -> Dict[str, Any]:
        """
        Analyze a predefined model (assuming generic GPU TFLOPS).
        
        Args:
            model_name: Name of the predefined model
            input_sequence_length: Input sequence length (uses model default if not specified)
            output_sequence_length: Desired output sequence length in tokens
            batch_size: Number of sequences processed in parallel
            precision: Data precision for inference
            gpu_tflops: GPU throughput in TFLOPS
            efficiency_factor: Computation efficiency factor (0-1)
            
        Returns:
            Dictionary with detailed analysis results, including keys like:
            - "model_info": Dictionary with model details (name, family, parameters_b, dimensions, etc.)
            - "analysis_parameters": Dictionary with input parameters used for analysis
            - "flops": Dictionary with FLOPs breakdown (attention, feedforward, prefill, per_token)
            - "vram": Dictionary with VRAM breakdown (weights, kv_cache, activations, total)
            - "performance": Dictionary with performance estimates (tokens_per_second, latency, etc.)
            - "overheads_used": Dictionary with overhead factors applied
            
        Raises:
            ValueError: If the model name is not recognized
        """
        # Get model configuration
        model_config = get_model_config(model_name)
        if not model_config:
            raise ValueError(f"Model '{model_name}' not recognized")
            
        # Extract model parameters
        hidden_dim = model_config["hidden_dimensions"]
        ff_dim = model_config["feedforward_dimensions"]
        num_layers = model_config["num_layers"]
        vocab_size = model_config["vocab_size"]
        
        # Use model-specific sequence length if not specified
        current_input_length = input_sequence_length
        if current_input_length is None:
            current_input_length = model_config.get("max_sequence_length", 2048)
        
        # Validate sequence lengths are positive
        if current_input_length <= 0:
            raise ValueError(f"Input sequence length must be positive, got {current_input_length}")
        if output_sequence_length <= 0:
             raise ValueError(f"Output sequence length must be positive, got {output_sequence_length}")
            
        # Calculate FLOPs (Prefill depends on input length)
        flops_attention = self.calculate_flops_attention(batch_size, current_input_length, hidden_dim)
        flops_feedforward = self.calculate_flops_feedforward(batch_size, current_input_length, hidden_dim, ff_dim)
        flops_prefill = self.calculate_flops_prefill(batch_size, current_input_length, hidden_dim, ff_dim, num_layers)
        flops_per_token = self.calculate_flops_per_token(batch_size, hidden_dim, ff_dim, num_layers)
        
        # Default overhead factors for VRAM calculations
        weights_overhead = 1.05
        kv_cache_overhead = 1.05
        activations_overhead = 1.1
        system_overhead = 1.05
        
        # Calculate VRAM requirements (Pass both lengths now)
        vram_dict = self.calculate_total_vram(
            batch_size=batch_size,
            input_sequence_length=current_input_length,
            hidden_dimensions=hidden_dim,
            feedforward_dimensions=ff_dim,
            num_layers=num_layers,
            vocab_size=vocab_size,
            output_sequence_length=output_sequence_length,
            precision=precision,
            weights_overhead=weights_overhead,
            kv_cache_overhead=kv_cache_overhead,
            activations_overhead=activations_overhead,
            system_overhead=system_overhead
        )
        
        # Calculate performance metrics (Latency depends on input/output lengths)
        tokens_per_second = self.estimate_inference_throughput(flops_per_token, gpu_tflops, efficiency_factor)
        prefill_latency = self.calculate_prefill_latency(flops_prefill, gpu_tflops, efficiency_factor)
        token_latency = self.calculate_token_latency(flops_per_token, gpu_tflops, efficiency_factor)
        time_for_1000_tokens = token_latency * 1000
        
        # Calculate TTFT and Total Request Time
        time_to_first_token = prefill_latency
        total_request_time = prefill_latency + (output_sequence_length * token_latency)
            
        # Calculate throughput on different GPUs (optional, might be heavy without specific GPU)
        throughput_by_gpu = self.calculate_throughput_across_gpus(flops_per_token, efficiency_factor)
        
        # Create result dictionary
        return {
            "model_info": {
                "name": model_name,
                "family": model_config.get("family", "Unknown"),
                "parameters_b": model_config.get("parameter_count", 0),
                "hidden_dimensions": hidden_dim,
                "feedforward_dimensions": ff_dim,
                "num_layers": num_layers,
                "vocab_size": vocab_size,
                "description": model_config.get("description", "")
            },
            "analysis_parameters": {
                "input_sequence_length": current_input_length,
                "output_sequence_length": output_sequence_length,
                "batch_size": batch_size,
                "precision": precision,
                "gpu_tflops": gpu_tflops,
                "efficiency_factor": efficiency_factor
            },
            "flops": {
                "attention": flops_attention,
                "feedforward": flops_feedforward,
                "prefill_total": flops_prefill,
                "per_token": flops_per_token
            },
            "vram": vram_dict,
            "performance": {
                "tokens_per_second": tokens_per_second,
                "prefill_latency": prefill_latency,
                "token_latency": token_latency,
                "time_to_first_token": time_to_first_token,
                "total_request_time": total_request_time,
                "time_for_1000_tokens": time_for_1000_tokens,
                "throughput_by_gpu": throughput_by_gpu
            },
             "overheads_used": {
                "weights": weights_overhead,
                "kv_cache": kv_cache_overhead,
                "activations": activations_overhead,
                "system": system_overhead
            }
        }
    
    def determine_model_scaling(self,
                          gpu_vram_gb: float,
                          interconnect_bandwidth_gb_per_sec: float,
                          batch_size: int,
                          sequence_length: int,
                          hidden_dimensions: int,
                          feedforward_dimensions: int,
                          num_layers: int,
                          vocab_size: int,
                          precision: Literal["fp16", "fp32", "bf16"] = "fp16") -> Dict[str, Any]:
        """
        Determine how a model can be scaled across GPUs based on VRAM requirements.
        
        Args:
            gpu_vram_gb: Available VRAM per GPU in GB
            interconnect_bandwidth_gb_per_sec: Interconnect bandwidth per second in GB
            batch_size: Number of sequences processed in parallel
            sequence_length: Sequence length for inference
            hidden_dimensions: Hidden size of the model
            feedforward_dimensions: Feedforward dimensions
            num_layers: Number of transformer layers
            vocab_size: Size of the vocabulary
            precision: Data precision for inference
            
        Returns:
            Dictionary with scaling analysis results including:
            - total_vram_required_gb: Total VRAM required for the model in GB
            - gpu_vram_capacity_gb: Available VRAM per GPU
            - fits_on_single_gpu: Whether the model fits on a single GPU
            - num_gpus_required: Estimated number of GPUs required
            - recommended_strategy: Recommended parallelism strategy
            - scaling_details: Details about different scaling options
        """
        # First calculate total VRAM requirements
        total_vram_result = self.calculate_total_vram(
            batch_size=batch_size,
            input_sequence_length=sequence_length,
            output_sequence_length=0,
            hidden_dimensions=hidden_dimensions,
            feedforward_dimensions=feedforward_dimensions,
            num_layers=num_layers,
            vocab_size=vocab_size,
            precision=precision
        )
        
        # Extract total VRAM requirement
        total_vram_required_gb = total_vram_result.get('total', 0)
        
        # Create model parameters dict for the VRAM calculator
        model_params = {
            'hidden_dimensions': hidden_dimensions,
            'feedforward_dimensions': feedforward_dimensions,
            'num_layers': num_layers,
            'vocab_size': vocab_size,
            'precision': precision
        }
        
        # Delegate to VRAM calculator's method
        scaling_result = self._vram.determine_model_scaling(
            gpu_vram_gb=gpu_vram_gb,
            interconnect_bandwidth_gb_per_sec=interconnect_bandwidth_gb_per_sec,
            total_vram_required_gb=total_vram_required_gb,
            model_params=model_params,
            num_layers=num_layers,
            hidden_dimensions=hidden_dimensions
        )
        
        # Add total VRAM required to the result
        scaling_result['total_vram_required_gb'] = total_vram_required_gb
        
        # Add to history
        if self._history:
            self._history.add_entry(
                f"Model Scaling Analysis:\n"
                f"  - Model size: {hidden_dimensions}-{feedforward_dimensions}-{num_layers}\n"
                f"  - Total VRAM required: {total_vram_required_gb:.2f} GB\n"
                f"  - GPU VRAM capacity: {gpu_vram_gb:.2f} GB\n"
                f"  - Fits on single GPU: {scaling_result.get('fits_on_single_gpu', False)}\n"
                f"  - GPUs required: {scaling_result.get('num_gpus_required', 'N/A')}\n"
                f"  - Recommended strategy: {scaling_result.get('recommended_strategy', 'N/A')}"
            )
        
        return scaling_result
    
    # Direct module methods with delegated implementation
    
    # FLOPs calculations
    def calculate_flops_attention(self, batch_size: int, sequence_length: int, hidden_dimensions: int) -> int:
        """Delegate to FLOPsCalculator.calculate_attention"""
        return self._flops.calculate_attention(batch_size, sequence_length, hidden_dimensions)
    
    def calculate_flops_feedforward(self, batch_size: int, sequence_length: int, 
                               hidden_dimensions: int, feedforward_dimensions: int) -> int:
        """Delegate to FLOPsCalculator.calculate_feedforward"""
        return self._flops.calculate_feedforward(batch_size, sequence_length, 
                                         hidden_dimensions, feedforward_dimensions)
    
    def calculate_flops_prefill(self, batch_size: int, sequence_length: int, 
                           hidden_dimensions: int, feedforward_dimensions: int, 
                           num_layers: int) -> int:
        """Delegate to FLOPsCalculator.calculate_prefill"""
        return self._flops.calculate_prefill(batch_size, sequence_length, 
                                    hidden_dimensions, feedforward_dimensions, num_layers)
    
    def calculate_flops_per_token(self, batch_size: int, hidden_dimensions: int,
                            feedforward_dimensions: int, num_layers: int) -> int:
        """Delegate to FLOPsCalculator.calculate_flops_per_token"""
        return self._flops.calculate_flops_per_token(batch_size, hidden_dimensions,
                                             feedforward_dimensions, num_layers)
    
    # VRAM calculations
    def calculate_model_vram(self, 
                        hidden_dimensions: int, 
                        feedforward_dimensions: int, 
                        num_layers: int,
                        vocab_size: int,
                        precision: Literal["fp16", "fp32", "bf16"] = "fp16",
                        overhead_factor: float = 1.0) -> float:
        """Delegate to VRAMCalculator.calculate_model_vram"""
        return self._vram.calculate_model_vram(
            hidden_dimensions, feedforward_dimensions, num_layers,
            vocab_size, precision, overhead_factor
        )
    
    def calculate_kv_cache_vram(self,
                          batch_size: int,
                          sequence_length: int,
                          hidden_dimensions: int,
                          num_layers: int,
                          precision: Literal["fp16", "fp32", "bf16"] = "fp16",
                          overhead_factor: float = 1.0) -> float:
        """Delegate to VRAMCalculator.calculate_kv_cache_vram"""
        return self._vram.calculate_kv_cache_vram(
            batch_size, sequence_length, hidden_dimensions,
            num_layers, precision, overhead_factor
        )
    
    def calculate_activations_vram(self,
                              batch_size: int,
                              sequence_length: int,
                              hidden_dimensions: int,
                              num_layers: int,
                              precision: Literal["fp16", "fp32", "bf16"] = "fp16",
                              overhead_factor: float = 1.1) -> float:
        """Delegate to VRAMCalculator.calculate_activations_vram"""
        return self._vram.calculate_activations_vram(
            batch_size, sequence_length, hidden_dimensions,
            num_layers, precision, overhead_factor
        )
    
    def calculate_total_vram(self,
                       batch_size: int,
                       input_sequence_length: int,
                       hidden_dimensions: int,
                       feedforward_dimensions: int,
                       num_layers: int,
                       vocab_size: int,
                       output_sequence_length: int = 0,
                       precision: Literal["fp16", "fp32", "bf16"] = "fp16",
                       weights_overhead: float = 1.05,
                       kv_cache_overhead: float = 1.05,
                       activations_overhead: float = 1.1,
                       system_overhead: float = 1.05) -> Dict[str, float]:
        """
        Calculate total VRAM required for inference including all components.
        Uses input_sequence_length for activations and the sum for KV cache peak.

        Args:
            batch_size: Number of sequences processed in parallel
            input_sequence_length: Sequence length of the input prompt
            hidden_dimensions: Hidden size of the model
            feedforward_dimensions: Feedforward dimensions
            num_layers: Number of transformer layers
            vocab_size: Size of the vocabulary
            output_sequence_length: Number of tokens to generate (defaults to 0)
            precision: Data precision for inference (defaults to "fp16")
            weights_overhead: Overhead factor for model weights (defaults to 1.05)
            kv_cache_overhead: Overhead factor for KV cache (defaults to 1.05)
            activations_overhead: Overhead factor for activations (defaults to 1.1)
            system_overhead: Overall system overhead factor (defaults to 1.05)

        Returns:
            Dictionary with detailed VRAM breakdown.
        """
        # Calculate peak sequence length for KV Cache
        total_sequence_length = input_sequence_length + output_sequence_length

        # Calculate BASE VRAM components
        model_vram_base = self._vram.calculate_model_vram_base(
            hidden_dimensions=hidden_dimensions,
            feedforward_dimensions=feedforward_dimensions,
            num_layers=num_layers,
            vocab_size=vocab_size,
            precision=precision
        )
        # Use TOTAL length for peak KV cache size
        kv_cache_vram_base = self._vram.calculate_kv_cache_vram_base(
            batch_size=batch_size,
            sequence_length=total_sequence_length, # Use sum here
            hidden_dimensions=hidden_dimensions,
            num_layers=num_layers,
            precision=precision
        )
        # Use INPUT length for peak activations size
        activations_vram_base = self._vram.calculate_activations_vram_base(
            batch_size=batch_size,
            sequence_length=input_sequence_length, # Use input length here
            hidden_dimensions=hidden_dimensions,
            num_layers=num_layers,
            precision=precision
        )

        # Calculate WITH OVERHEAD VRAM components
        model_vram_with_overhead = model_vram_base * weights_overhead
        kv_cache_vram_with_overhead = kv_cache_vram_base * kv_cache_overhead
        activations_vram_with_overhead = activations_vram_base * activations_overhead

        # Calculate totals
        total_vram_base = model_vram_base + kv_cache_vram_base + activations_vram_base
        component_subtotal_with_overhead = (
            model_vram_with_overhead + kv_cache_vram_with_overhead + activations_vram_with_overhead
        )
        total_vram_system_wide = component_subtotal_with_overhead * system_overhead

        result = {
            # Base values
            "weights_base": model_vram_base,
            "kv_cache_base": kv_cache_vram_base,
            "activations_base": activations_vram_base,
            "total_base": total_vram_base,

            # With component overhead
            "weights_with_overhead": model_vram_with_overhead,
            "kv_cache_with_overhead": kv_cache_vram_with_overhead,
            "activations_with_overhead": activations_vram_with_overhead,
            "component_subtotal": component_subtotal_with_overhead,

            # With system-wide overhead
            "system_overhead_applied": total_vram_system_wide - component_subtotal_with_overhead,
            "total": total_vram_system_wide
        }

        if self._history:
             self._history.add_entry(
                f"Total_VRAM_Details:\n"
                f"  - Input Len={input_sequence_length}, Output Len={output_sequence_length}, Total Len={total_sequence_length}\n"
                f"  - Base: Model={model_vram_base:.2f}, KV={kv_cache_vram_base:.2f}, Act={activations_vram_base:.2f}, Total={total_vram_base:.2f} GB\n"
                f"  - +Overhead: Model={model_vram_with_overhead:.2f}, KV={kv_cache_vram_with_overhead:.2f}, Act={activations_vram_with_overhead:.2f}, Subtotal={component_subtotal_with_overhead:.2f} GB\n"
                f"  - +System Overhead: {result['system_overhead_applied']:.2f} GB\n"
                f"  - Final Total: {total_vram_system_wide:.2f} GB"
            )

        return result
    
    # Throughput calculations
    def estimate_inference_throughput(self, 
                                 flops_per_token: int, 
                                 gpu_tflops: float, 
                                 efficiency_factor: float = 0.3) -> float:
        """Delegate to ThroughputCalculator.estimate_inference_throughput"""
        return self._throughput.estimate_inference_throughput(
            flops_per_token, gpu_tflops, efficiency_factor
        )
    
    def estimate_batch_throughput(self,
                             batch_size: int,
                             flops_per_token: int,
                             gpu_tflops: float,
                             num_gpus: int = 1,
                             parallel_efficiency: float = 0.9,
                             compute_efficiency: float = 0.3) -> Dict[str, Any]:
        """Delegate to ThroughputCalculator.estimate_batch_throughput"""
        return self._throughput.estimate_batch_throughput(
            batch_size, flops_per_token, gpu_tflops,
            num_gpus, parallel_efficiency, compute_efficiency
        )
    
    def calculate_throughput_across_gpus(self,
                                      flops_per_token: int,
                                      efficiency_factor: float = 0.3) -> Dict[str, float]:
        """Delegate to ThroughputCalculator.calculate_throughput_across_gpus"""
        return self._throughput.calculate_throughput_across_gpus(
            flops_per_token, efficiency_factor
        )
    
    # Latency calculations
    def calculate_prefill_latency(self,
                               flops_prefill: int,
                               gpu_tflops: float,
                               efficiency_factor: float = 0.3) -> float:
        """Delegate to LatencyCalculator.calculate_prefill_latency"""
        return self._latency.calculate_prefill_latency(
            flops_prefill, gpu_tflops, efficiency_factor
        )
    
    def calculate_token_latency(self,
                             flops_per_token: int,
                             gpu_tflops: float,
                             efficiency_factor: float = 0.3) -> float:
        """Delegate to LatencyCalculator.calculate_token_latency"""
        return self._latency.calculate_token_latency(
            flops_per_token, gpu_tflops, efficiency_factor
        )
    
    def calculate_completion_latency(self,
                                  prompt_length: int,
                                  output_length: int,
                                  flops_prefill: int,
                                  flops_per_token: int,
                                  gpu_tflops: float,
                                  efficiency_factor: float = 0.3) -> Dict[str, Any]:
        """Delegate to LatencyCalculator.calculate_completion_latency"""
        return self._latency.calculate_completion_latency(
            prompt_length, output_length, flops_prefill,
            flops_per_token, gpu_tflops, efficiency_factor
        )
    
    # Aliases for backward compatibility
    
    # FLOPs aliases
    calculate_flops_per_token = calculate_flops_per_token
    
    # Throughput aliases
    calculate_tokens_per_second = estimate_inference_throughput
    
    # Latency aliases
    estimate_prefill_latency = calculate_prefill_latency
    estimate_token_generation_latency = calculate_token_latency
    estimate_completion_latency = calculate_completion_latency
    
    # History management
    def clear_history(self) -> None:
        """Clear calculation history."""
        self._history.clear()
    
    def get_history(self) -> List[str]:
        """Get calculation history as a list of entries."""
        return self._history.get_entries()
