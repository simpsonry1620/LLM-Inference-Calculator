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
            
            # Calculate VRAM for model weights
            model_vram_gb = self.calculate_model_vram(
                hidden_dim, ff_dim, num_layers, vocab_size, precision="fp16"
            )
        else:
            # Direct VRAM specification
            model_vram_gb = float(model_name_or_vram)
        
        # Get recommended GPUs
        return get_recommended_gpu_for_model(model_vram_gb, min_vram_headroom_gb)
    
    def analyze_model_on_gpu(self, 
                           model_name: str, 
                           gpu_name: str,
                           sequence_length: Optional[int] = None,
                           batch_size: int = 1,
                           precision: Literal["fp16", "fp32", "bf16", "int8", "int4"] = "fp16",
                           efficiency_factor: float = 0.3) -> Dict[str, Any]:
        """
        Analyze how a specific model would perform on a specific GPU.
        
        Args:
            model_name: Name of the predefined model
            gpu_name: Name of the GPU to analyze on
            sequence_length: Custom sequence length (uses model default if None)
            batch_size: Batch size for inference
            precision: Numerical precision for the model
            efficiency_factor: Efficiency factor for throughput and latency calculations
            
        Returns:
            Dictionary with comprehensive model-on-GPU analysis
            
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
        
        # Extract model parameters
        hidden_dim = model_config["hidden_dimensions"]
        ff_dim = model_config["feedforward_dimensions"]
        num_layers = model_config["num_layers"]
        vocab_size = model_config["vocab_size"]
        seq_len = sequence_length if sequence_length is not None else model_config["default_seq_length"]
        
        # Extract GPU parameters
        gpu_vram_gb = gpu_config["vram_gb"]
        
        # Get appropriate TFLOPS for the selected precision
        if precision == "fp32":
            gpu_tflops = gpu_config["fp32_tflops"]
        elif precision in ["fp16", "bf16"]:
            gpu_tflops = gpu_config.get(f"{precision}_tflops", gpu_config["fp16_tflops"])
        elif precision == "int8":
            gpu_tflops = gpu_config.get("int8_tflops", gpu_config["fp16_tflops"] * 2)
        elif precision == "int4":
            gpu_tflops = gpu_config.get("int4_tflops", gpu_config["fp16_tflops"] * 4)
        else:
            # Default to fp16 if precision not found
            gpu_tflops = gpu_config["fp16_tflops"]
        
        # Check if GPU supports the selected precision
        supported_precisions = [p.lower() for p in gpu_config["supported_precisions"]]
        precision_supported = precision.lower() in supported_precisions
        
        # Calculate model VRAM requirements
        model_vram = self.calculate_model_vram(
            hidden_dim, ff_dim, num_layers, vocab_size, precision
        )
        
        kv_cache_vram = self.calculate_kv_cache_vram(
            batch_size, seq_len, hidden_dim, num_layers, precision
        )
        
        total_vram = self.calculate_total_vram(
            batch_size, seq_len, hidden_dim, ff_dim, num_layers, vocab_size, precision
        )
        
        # Determine if the model fits on the GPU
        vram_fits = total_vram <= gpu_vram_gb
        
        # Calculate FLOPs
        flops_prefill = self.calculate_flops_prefill(batch_size, seq_len, hidden_dim, ff_dim, num_layers)
        
        # Calculate per-token FLOPs (for a single new token)
        flops_per_token = (
            self.calculate_flops_attention(batch_size, 1, hidden_dim) +
            self.calculate_flops_feedforward(batch_size, 1, hidden_dim, ff_dim)
        ) * num_layers
        
        # Throughput and latency with the specific GPU
        tokens_per_second = self.estimate_inference_throughput(
            flops_per_token, gpu_tflops, efficiency_factor
        )
        
        prefill_latency = self.estimate_prefill_latency(
            flops_prefill, gpu_tflops, efficiency_factor
        )
        
        token_latency = self.estimate_token_generation_latency(
            flops_per_token, gpu_tflops, efficiency_factor
        )
        
        # Memory bandwidth utilization estimation (simplified)
        # Assuming each token processing reads/writes ~3x the model size in data
        bandwidth_required = 3 * model_vram * tokens_per_second  # GB/s
        bandwidth_utilization = min(1.0, bandwidth_required / gpu_config["bandwidth_gb_per_sec"])
        
        # Return comprehensive analysis
        return {
            "model_info": model_config,
            "gpu_info": gpu_config,
            "analysis_parameters": {
                "sequence_length": seq_len,
                "batch_size": batch_size,
                "precision": precision,
                "precision_supported": precision_supported,
                "efficiency_factor": efficiency_factor
            },
            "compatibility": {
                "vram_required": total_vram,
                "vram_available": gpu_vram_gb,
                "vram_fits": vram_fits,
                "vram_headroom_gb": max(0, gpu_vram_gb - total_vram) if vram_fits else 0,
                "maximum_batch_size": min(
                    gpu_config["max_batch_size"],
                    int((gpu_vram_gb - model_vram) / kv_cache_vram) if kv_cache_vram > 0 else float('inf')
                ),
                "maximum_sequence_length": int(
                    seq_len * (gpu_vram_gb - model_vram) / kv_cache_vram
                ) if kv_cache_vram > 0 and vram_fits else 0
            },
            "performance": {
                "tokens_per_second": tokens_per_second,
                "prefill_latency": prefill_latency,
                "token_latency": token_latency,
                "time_for_1000_tokens": token_latency * 1000,
                "bandwidth_utilization": bandwidth_utilization
            }
        }
    
    def analyze_model_by_name(self, 
                             model_name: str, 
                             sequence_length: Optional[int] = None,
                             batch_size: int = 1,
                             precision: Literal["fp16", "fp32", "bf16"] = "fp16",
                             gpu_tflops: float = 312.0,  # A100 defaults
                             efficiency_factor: float = 0.3) -> Dict[str, Any]:
        """
        Perform a comprehensive analysis of a predefined model.
        
        Args:
            model_name: Name of the predefined model to analyze
            sequence_length: Custom sequence length (uses model default if None)
            batch_size: Batch size for inference
            precision: Numerical precision for the model
            gpu_tflops: GPU throughput in TFLOPS
            efficiency_factor: Efficiency factor for throughput and latency calculations
            
        Returns:
            Dictionary with comprehensive model analysis results,
            or None if the model was not found
            
        Raises:
            ValueError: If the model name is not recognized
        """
        # Get model configuration
        model_config = get_model_config(model_name)
        if not model_config:
            raise ValueError(f"Model '{model_name}' not recognized. Use get_available_models() to see options.")
        
        # Extract model parameters
        hidden_dim = model_config["hidden_dimensions"]
        ff_dim = model_config["feedforward_dimensions"]
        num_layers = model_config["num_layers"]
        vocab_size = model_config["vocab_size"]
        seq_len = sequence_length if sequence_length is not None else model_config["default_seq_length"]
        
        # FLOPs calculations
        flops_attention = self.calculate_flops_attention(batch_size, seq_len, hidden_dim)
        flops_feedforward = self.calculate_flops_feedforward(batch_size, seq_len, hidden_dim, ff_dim)
        flops_prefill = self.calculate_flops_prefill(batch_size, seq_len, hidden_dim, ff_dim, num_layers)
        
        # Calculate per-token FLOPs (for a single new token)
        flops_per_token = (
            self.calculate_flops_attention(batch_size, 1, hidden_dim) +
            self.calculate_flops_feedforward(batch_size, 1, hidden_dim, ff_dim)
        ) * num_layers
        
        # VRAM calculations
        model_vram = self.calculate_model_vram(
            hidden_dim, ff_dim, num_layers, vocab_size, precision
        )
        
        kv_cache_vram = self.calculate_kv_cache_vram(
            batch_size, seq_len, hidden_dim, num_layers, precision
        )
        
        total_vram = self.calculate_total_vram(
            batch_size, seq_len, hidden_dim, ff_dim, num_layers, vocab_size, precision
        )
        
        # Throughput and latency
        tokens_per_second = self.estimate_inference_throughput(
            flops_per_token, gpu_tflops, efficiency_factor
        )
        
        prefill_latency = self.estimate_prefill_latency(
            flops_prefill, gpu_tflops, efficiency_factor
        )
        
        token_latency = self.estimate_token_generation_latency(
            flops_per_token, gpu_tflops, efficiency_factor
        )
        
        # Compare vs lower and higher-end GPUS
        throughput_values = {
            "A100": self.estimate_inference_throughput(flops_per_token, 312.0, efficiency_factor),
            "H100": self.estimate_inference_throughput(flops_per_token, 756.0, efficiency_factor),
            "H200": self.estimate_inference_throughput(flops_per_token, 989.0, efficiency_factor),
            "RTX 4090": self.estimate_inference_throughput(flops_per_token, 82.6, efficiency_factor),
            "RTX 3090": self.estimate_inference_throughput(flops_per_token, 35.6, efficiency_factor)
        }
        
        # Return comprehensive results
        return {
            "model_info": model_config,
            "analysis_parameters": {
                "sequence_length": seq_len,
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
            "vram": {
                "model_weights": model_vram,
                "kv_cache": kv_cache_vram,
                "total": total_vram
            },
            "performance": {
                "tokens_per_second": tokens_per_second,
                "prefill_latency": prefill_latency,
                "token_latency": token_latency,
                "time_for_1000_tokens": token_latency * 1000,
                "throughput_by_gpu": throughput_values
            }
        }
    
    # FLOPs calculation methods
    def calculate_flops_attention(self, batch_size: int, sequence_length: int, hidden_dimensions: int) -> int:
        """
        Calculate FLOPs for attention mechanism.
        
        Args:
            batch_size: Number of sequences processed in parallel
            sequence_length: Length of input sequences in tokens
            hidden_dimensions: Size of hidden dimensions in the model
            
        Returns:
            Estimated FLOPs for attention computation
        """
        return self._flops.calculate_attention(batch_size, sequence_length, hidden_dimensions)
    
    def calculate_flops_feedforward(self, batch_size: int, sequence_length: int, 
                                   hidden_dimensions: int, feedforward_dimensions: int) -> int:
        """
        Calculate FLOPs for feedforward operations.
        
        Args:
            batch_size: Number of sequences processed in parallel
            sequence_length: Length of input sequences in tokens
            hidden_dimensions: Size of hidden dimensions in the model
            feedforward_dimensions: Size of feedforward network dimensions
            
        Returns:
            Estimated FLOPs for feedforward computation
        """
        return self._flops.calculate_feedforward(
            batch_size, sequence_length, hidden_dimensions, feedforward_dimensions
        )
    
    def calculate_flops_prefill(self, batch_size: int, sequence_length: int, 
                               hidden_dimensions: int, feedforward_dimensions: int, 
                               num_layers: int) -> int:
        """
        Calculate total FLOPs for prefill phase (processing full context).
        
        Args:
            batch_size: Number of sequences processed in parallel
            sequence_length: Length of input sequences in tokens
            hidden_dimensions: Size of hidden dimensions in the model
            feedforward_dimensions: Size of feedforward network dimensions
            num_layers: Number of transformer layers
            
        Returns:
            Estimated total FLOPs for prefill computation
        """
        return self._flops.calculate_prefill(
            batch_size, sequence_length, hidden_dimensions, feedforward_dimensions, num_layers
        )
    
    # VRAM calculation methods
    def calculate_model_vram(self, 
                            hidden_dimensions: int, 
                            feedforward_dimensions: int, 
                            num_layers: int,
                            vocab_size: int,
                            precision: Literal["fp16", "fp32", "bf16"] = "fp16",
                            overhead_factor: float = 1.0) -> float:
        """
        Calculate VRAM required for model weights.
        
        Args:
            hidden_dimensions: Size of hidden dimensions in the model
            feedforward_dimensions: Size of feedforward network dimensions
            num_layers: Number of transformer layers
            vocab_size: Size of vocabulary for embeddings
            precision: Data precision (fp16=2 bytes, bf16=2 bytes, fp32=4 bytes)
            overhead_factor: Multiplicative factor to account for runtime overhead
            
        Returns:
            VRAM size in gigabytes (GB)
        """
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
        """
        Calculate VRAM required for KV (Key-Value) cache during inference.
        
        Args:
            batch_size: Number of sequences processed in parallel
            sequence_length: Length of input sequences in tokens
            hidden_dimensions: Size of hidden dimensions in the model
            num_layers: Number of transformer layers
            precision: Data precision
            overhead_factor: Multiplicative factor for memory alignment and padding
            
        Returns:
            KV cache VRAM size in gigabytes (GB)
        """
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
        """
        Calculate peak VRAM required for activations during inference.
        
        Args:
            batch_size: Number of sequences processed in parallel
            sequence_length: Length of input sequences in tokens
            hidden_dimensions: Size of hidden dimensions in the model
            num_layers: Number of transformer layers
            precision: Data precision
            overhead_factor: Multiplicative factor to account for runtime overhead
            
        Returns:
            Peak activations VRAM size in gigabytes (GB)
        """
        return self._vram.calculate_activations_vram(
            batch_size, sequence_length, hidden_dimensions, 
            num_layers, precision, overhead_factor
        )
    
    def calculate_total_vram(self,
                           batch_size: int,
                           sequence_length: int,
                           hidden_dimensions: int,
                           feedforward_dimensions: int,
                           num_layers: int,
                           vocab_size: int,
                           precision: Literal["fp16", "fp32", "bf16"] = "fp16",
                           weights_overhead: float = 1.05,
                           kv_cache_overhead: float = 1.05,
                           activations_overhead: float = 1.1,
                           system_overhead: float = 1.05) -> float:
        """
        Calculate total VRAM required for model inference.
        
        Args:
            batch_size: Number of sequences processed in parallel
            sequence_length: Length of input sequences in tokens
            hidden_dimensions: Size of hidden dimensions in the model
            feedforward_dimensions: Size of feedforward network dimensions
            num_layers: Number of transformer layers
            vocab_size: Size of vocabulary for embeddings
            precision: Data precision
            weights_overhead, kv_cache_overhead, activations_overhead, system_overhead:
                Overhead factors for different components
            
        Returns:
            Total VRAM size in gigabytes (GB)
        """
        return self._vram.calculate_total_vram(
            batch_size, sequence_length, hidden_dimensions, feedforward_dimensions,
            num_layers, vocab_size, precision, weights_overhead, kv_cache_overhead,
            activations_overhead, system_overhead
        )
    
    # Throughput calculation methods
    def estimate_inference_throughput(self, 
                                     flops_per_token: int, 
                                     gpu_tflops: float, 
                                     efficiency_factor: float = 0.3) -> float:
        """
        Estimate inference throughput in tokens per second.
        
        Args:
            flops_per_token: Number of FLOPs required to generate a token
            gpu_tflops: GPU throughput in TFLOPS (fp16/bf16)
            efficiency_factor: Efficiency factor for real-world performance
            
        Returns:
            Estimated tokens per second
        """
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
        """
        Estimate batch processing throughput for multiple GPUs.
        
        Args:
            batch_size: Batch size for processing
            flops_per_token: FLOPs required per token
            gpu_tflops: Single GPU throughput in TFLOPS
            num_gpus: Number of GPUs used for inference
            parallel_efficiency: Efficiency of parallelization (0-1)
            compute_efficiency: GPU compute efficiency (0-1)
            
        Returns:
            Dictionary with throughput metrics
        """
        return self._throughput.estimate_batch_throughput(
            batch_size, flops_per_token, gpu_tflops, 
            num_gpus, parallel_efficiency, compute_efficiency
        )
    
    # Latency calculation methods
    def estimate_prefill_latency(self,
                               flops_prefill: int,
                               gpu_tflops: float,
                               efficiency_factor: float = 0.3) -> float:
        """
        Estimate prefill phase latency in seconds.
        
        Args:
            flops_prefill: Total FLOPs for the prefill phase
            gpu_tflops: GPU throughput in TFLOPS (fp16/bf16)
            efficiency_factor: Efficiency factor for real-world performance
            
        Returns:
            Estimated prefill latency in seconds
        """
        return self._latency.estimate_prefill_latency(
            flops_prefill, gpu_tflops, efficiency_factor
        )
    
    def estimate_token_generation_latency(self,
                                        flops_per_token: int,
                                        gpu_tflops: float,
                                        efficiency_factor: float = 0.3) -> float:
        """
        Estimate per-token generation latency in seconds.
        
        Args:
            flops_per_token: FLOPs required to generate a single token
            gpu_tflops: GPU throughput in TFLOPS (fp16/bf16)
            efficiency_factor: Efficiency factor for real-world performance
            
        Returns:
            Estimated latency per token in seconds
        """
        return self._latency.estimate_token_generation_latency(
            flops_per_token, gpu_tflops, efficiency_factor
        )
    
    def estimate_completion_latency(self,
                                  prompt_length: int,
                                  output_length: int,
                                  flops_prefill: int,
                                  flops_per_token: int,
                                  gpu_tflops: float,
                                  efficiency_factor: float = 0.3) -> Dict[str, Any]:
        """
        Estimate end-to-end completion latency including prefill and token generation.
        
        Args:
            prompt_length: Length of input prompt in tokens
            output_length: Expected output length in tokens
            flops_prefill: Total FLOPs for prefill phase
            flops_per_token: FLOPs per token for generation phase
            gpu_tflops: GPU throughput in TFLOPS
            efficiency_factor: Efficiency factor for calculations
            
        Returns:
            Dictionary with latency metrics
        """
        return self._latency.estimate_completion_latency(
            prompt_length, output_length, flops_prefill, 
            flops_per_token, gpu_tflops, efficiency_factor
        )
    
    # History management methods
    def clear_history(self) -> None:
        """Clear calculation history"""
        self._history.clear()
    
    def get_history(self) -> List[str]:
        """Return calculation history"""
        return self._history.get_entries()
