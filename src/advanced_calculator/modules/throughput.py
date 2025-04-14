from typing import Optional, Dict, Any
from src.advanced_calculator.modules.utils import validate_positive_integer, validate_positive_number

class ThroughputCalculator:
    """
    Calculator for estimating throughput of LLM inference and training.
    """
    def __init__(self, history_callback=None):
        """
        Initialize throughput calculator
        
        Args:
            history_callback: Optional callback function to log calculation history
        """
        self._history_callback = history_callback
        
    def estimate_inference_throughput(self, 
                                     flops_per_token: int, 
                                     gpu_tflops: float, 
                                     efficiency_factor: float = 0.3) -> float:
        """
        Estimate inference throughput in tokens per second.
        
        Args:
            flops_per_token: Number of FLOPs required to generate a token
            gpu_tflops: GPU throughput in TFLOPS (fp16/bf16)
            efficiency_factor: Efficiency factor (0-1) for real-world performance vs. theoretical (default: 0.3)
            
        Returns:
            Estimated tokens per second
        """
        validate_positive_integer(flops_per_token, "FLOPs per token")
        validate_positive_number(gpu_tflops, "GPU TFLOPS")
        validate_positive_number(efficiency_factor, "Efficiency factor")
        
        if efficiency_factor > 1.0:
            raise ValueError(f"Efficiency factor must be <= 1.0, got {efficiency_factor}")
        
        # Convert GPU TFLOPS to FLOPS
        gpu_flops = gpu_tflops * (10 ** 12)
        
        # Calculate theoretical throughput
        theoretical_tokens_per_second = gpu_flops / flops_per_token
        
        # Apply efficiency factor for realistic estimate
        realistic_tokens_per_second = theoretical_tokens_per_second * efficiency_factor
        
        if self._history_callback:
            self._history_callback(
                f"Inference_Throughput(flops_per_token={flops_per_token}, "
                f"gpu_tflops={gpu_tflops}, efficiency_factor={efficiency_factor}) = "
                f"{realistic_tokens_per_second:.2f} tokens/second"
            )
        
        return realistic_tokens_per_second
    
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
        validate_positive_integer(batch_size, "Batch size")
        validate_positive_integer(flops_per_token, "FLOPs per token")
        validate_positive_number(gpu_tflops, "GPU TFLOPS")
        validate_positive_integer(num_gpus, "Number of GPUs")
        validate_positive_number(parallel_efficiency, "Parallel efficiency")
        validate_positive_number(compute_efficiency, "Compute efficiency")
        
        if parallel_efficiency > 1.0 or compute_efficiency > 1.0:
            raise ValueError("Efficiency values must be <= 1.0")
        
        # Calculate single GPU throughput
        single_gpu_tokens_per_second = self.estimate_inference_throughput(
            flops_per_token, gpu_tflops, compute_efficiency
        )
        
        # Apply parallelization efficiency for multiple GPUs
        if num_gpus > 1:
            # Simplified scaling model: efficiency decreases with more GPUs
            multi_gpu_scaling = num_gpus * parallel_efficiency
        else:
            multi_gpu_scaling = 1.0
            
        total_tokens_per_second = single_gpu_tokens_per_second * multi_gpu_scaling
        
        # Calculate batch throughput
        batch_throughput = total_tokens_per_second / batch_size
        batches_per_second = total_tokens_per_second / batch_size
        
        result = {
            "tokens_per_second": total_tokens_per_second,
            "batches_per_second": batches_per_second,
            "tokens_per_second_per_gpu": single_gpu_tokens_per_second,
            "effective_gpu_scaling": multi_gpu_scaling
        }
        
        if self._history_callback:
            self._history_callback(
                f"Batch_Throughput(batch_size={batch_size}, num_gpus={num_gpus}, "
                f"parallel_efficiency={parallel_efficiency}, compute_efficiency={compute_efficiency}) = "
                f"{batches_per_second:.2f} batches/second, {total_tokens_per_second:.2f} tokens/second"
            )
        
        return result
