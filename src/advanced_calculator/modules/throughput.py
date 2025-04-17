from typing import Optional, Dict, Any, List
from src.advanced_calculator.modules.utils import validate_positive_integer, validate_positive_number
from src.advanced_calculator.modules.gpus import list_all_gpus, get_gpu_config

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
    
    def estimate_prefill_latency(self,
                               flops_prefill: int,
                               gpu_tflops: float,
                               efficiency_factor: float = 0.3) -> float:
        """
        Estimate the latency for the prefill phase of inference.
        
        Args:
            flops_prefill: Total FLOPs required for the prefill phase
            gpu_tflops: GPU throughput in TFLOPS
            efficiency_factor: Efficiency factor (0-1) for real-world performance
            
        Returns:
            Estimated prefill latency in seconds
        """
        validate_positive_integer(flops_prefill, "FLOPs prefill")
        validate_positive_number(gpu_tflops, "GPU TFLOPS")
        validate_positive_number(efficiency_factor, "Efficiency factor")
        
        if efficiency_factor > 1.0:
            raise ValueError(f"Efficiency factor must be <= 1.0, got {efficiency_factor}")
        
        # Convert GPU TFLOPS to FLOPS
        gpu_flops = gpu_tflops * (10 ** 12)
        
        # Calculate theoretical time required
        theoretical_time = flops_prefill / gpu_flops
        
        # Apply efficiency factor for realistic estimate
        realistic_time = theoretical_time / efficiency_factor
        
        if self._history_callback:
            self._history_callback(
                f"Prefill_Latency(flops_prefill={flops_prefill}, "
                f"gpu_tflops={gpu_tflops}, efficiency_factor={efficiency_factor}) = "
                f"{realistic_time:.4f} seconds"
            )
        
        return realistic_time
    
    def estimate_token_generation_latency(self,
                                        flops_per_token: int,
                                        gpu_tflops: float,
                                        efficiency_factor: float = 0.3) -> float:
        """
        Estimate the latency for generating a single token.
        
        Args:
            flops_per_token: FLOPs required to generate a single token
            gpu_tflops: GPU throughput in TFLOPS
            efficiency_factor: Efficiency factor (0-1) for real-world performance
            
        Returns:
            Estimated token generation latency in seconds
        """
        # The token generation latency is simply the inverse of throughput
        tokens_per_second = self.estimate_inference_throughput(
            flops_per_token, gpu_tflops, efficiency_factor
        )
        
        latency = 1.0 / tokens_per_second
        
        if self._history_callback:
            self._history_callback(
                f"Token_Generation_Latency(flops_per_token={flops_per_token}, "
                f"gpu_tflops={gpu_tflops}, efficiency_factor={efficiency_factor}) = "
                f"{latency:.6f} seconds"
            )
        
        return latency
    
    def estimate_batch_throughput(self,
                                 batch_size: int,
                                 flops_per_token: int,
                                 gpu_tflops: float,
                                 num_gpus: int = 1,
                                 interconnect_bandwidth_gb_per_sec: Optional[float] = None,
                                 compute_efficiency: float = 0.3) -> Dict[str, Any]:
        """
        Estimate batch processing throughput for multiple GPUs.
        
        Args:
            batch_size: Batch size for processing
            flops_per_token: FLOPs required per token
            gpu_tflops: Single GPU throughput in TFLOPS
            num_gpus: Number of GPUs used for inference
            interconnect_bandwidth_gb_per_sec: Interconnect bandwidth (used if num_gpus > 1)
            compute_efficiency: GPU compute efficiency (0-1)
            
        Returns:
            Dictionary with throughput metrics
        """
        validate_positive_integer(batch_size, "Batch size")
        validate_positive_integer(flops_per_token, "FLOPs per token")
        validate_positive_number(gpu_tflops, "GPU TFLOPS")
        validate_positive_integer(num_gpus, "Number of GPUs")
        validate_positive_number(compute_efficiency, "Compute efficiency")
        
        if compute_efficiency > 1.0:
            raise ValueError("Compute efficiency value must be <= 1.0")
        
        # Calculate single GPU throughput
        single_gpu_tokens_per_second = self.estimate_inference_throughput(
            flops_per_token, gpu_tflops, compute_efficiency
        )
        
        # Apply parallelization efficiency for multiple GPUs
        if num_gpus > 1:
            if interconnect_bandwidth_gb_per_sec is None:
                raise ValueError("Interconnect bandwidth must be provided for multi-GPU throughput estimation.")
            
            # Calculate parallel efficiency based on bandwidth and number of GPUs
            if interconnect_bandwidth_gb_per_sec >= 600: # High (NVLink >= A100)
                base_efficiency = 0.95
            elif interconnect_bandwidth_gb_per_sec > 64: # Medium (PCIe 5+?)
                base_efficiency = 0.85
            else: # Low (PCIe <= 4)
                base_efficiency = 0.75

            efficiency_decay_per_gpu = 0.01 # Small efficiency drop per additional GPU
            calculated_parallel_efficiency = max(0.4, base_efficiency - efficiency_decay_per_gpu * (num_gpus - 1))
            
            multi_gpu_scaling = num_gpus * calculated_parallel_efficiency
            
            # Log the calculated efficiency
            if self._history_callback:
                self._history_callback(
                    f"  - Calculated parallel efficiency for {num_gpus} GPUs with {interconnect_bandwidth_gb_per_sec} GB/s BW: {calculated_parallel_efficiency:.3f}"
                )
        else:
            multi_gpu_scaling = 1.0
            calculated_parallel_efficiency = 1.0 # Set for logging consistency
            
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
                f"interconnect_bw={interconnect_bandwidth_gb_per_sec or 'N/A'} GB/s, calc_parallel_eff={calculated_parallel_efficiency:.3f}, compute_efficiency={compute_efficiency:.2f}) = "
                f"{batches_per_second:.2f} batches/second, {total_tokens_per_second:.2f} tokens/second"
            )
        
        return result
        
    def calculate_throughput_across_gpus(self,
                                       flops_per_token: int,
                                       efficiency_factor: float = 0.3) -> Dict[str, float]:
        """
        Calculate inference throughput across different GPU models.
        
        Args:
            flops_per_token: FLOPs required to generate a single token
            efficiency_factor: Efficiency factor for computation (0-1)
            
        Returns:
            Dictionary mapping GPU names to estimated throughput in tokens/second
        """
        validate_positive_integer(flops_per_token, "FLOPs per token")
        validate_positive_number(efficiency_factor, "Efficiency factor")
        
        results = {}
        
        # Get all available GPUs
        gpu_names = list_all_gpus()
        
        # Calculate throughput for each GPU
        for gpu_name in gpu_names:
            gpu_config = get_gpu_config(gpu_name)
            
            if not gpu_config:
                continue
                
            # Get TFLOPS based on the GPU architecture
            # Prefer fp16 performance, fallback to others
            tflops = gpu_config.get('fp16_tflops', 
                     gpu_config.get('bf16_tflops',
                     gpu_config.get('fp32_tflops', 0)))
            
            if tflops == 0:
                continue
                
            # Calculate throughput
            throughput = self.estimate_inference_throughput(
                flops_per_token, tflops, efficiency_factor
            )
            
            results[gpu_name] = throughput
        
        if self._history_callback:
            message = "Throughput_Across_GPUs:\n"
            for gpu, tps in results.items():
                message += f"  - {gpu}: {tps:.2f} tokens/second\n"
            self._history_callback(message)
            
        return results
