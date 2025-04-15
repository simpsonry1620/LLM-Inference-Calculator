from typing import Dict, Any, Optional
from src.advanced_calculator.modules.utils import validate_positive_integer, validate_positive_number

class LatencyCalculator:
    """
    Calculator for estimating latency of LLM operations.
    """
    def __init__(self, history_callback=None):
        """
        Initialize latency calculator
        
        Args:
            history_callback: Optional callback function to log calculation history
        """
        self._history_callback = history_callback
    
    def calculate_prefill_latency(self,
                               flops_prefill: int,
                               gpu_tflops: float,
                               efficiency_factor: float = 0.3) -> float:
        """
        Estimate prefill phase latency in seconds.
        
        The prefill phase processes the entire prompt/context before generation starts.
        
        Args:
            flops_prefill: Total FLOPs for the prefill phase
            gpu_tflops: GPU throughput in TFLOPS (fp16/bf16)
            efficiency_factor: Efficiency factor (0-1) for real-world performance (default: 0.3)
            
        Returns:
            Estimated prefill latency in seconds
        """
        validate_positive_integer(flops_prefill, "Prefill FLOPs")
        validate_positive_number(gpu_tflops, "GPU TFLOPS")
        validate_positive_number(efficiency_factor, "Efficiency factor")
        
        if efficiency_factor > 1.0:
            raise ValueError(f"Efficiency factor must be <= 1.0, got {efficiency_factor}")
        
        # Convert GPU TFLOPS to FLOPS
        gpu_flops = gpu_tflops * (10 ** 12)
        
        # Calculate theoretical latency
        theoretical_latency = flops_prefill / gpu_flops
        
        # Apply efficiency factor for realistic estimate
        realistic_latency = theoretical_latency / efficiency_factor
        
        if self._history_callback:
            self._history_callback(
                f"Prefill_Latency(flops_prefill={flops_prefill}, "
                f"gpu_tflops={gpu_tflops}, efficiency_factor={efficiency_factor}) = "
                f"{realistic_latency:.4f} seconds"
            )
        
        return realistic_latency
    
    # Alias for backward compatibility
    estimate_prefill_latency = calculate_prefill_latency
    
    def calculate_token_latency(self,
                             flops_per_token: int,
                             gpu_tflops: float,
                             efficiency_factor: float = 0.3) -> float:
        """
        Estimate per-token generation latency in seconds.
        
        Args:
            flops_per_token: FLOPs required to generate a single token
            gpu_tflops: GPU throughput in TFLOPS (fp16/bf16)
            efficiency_factor: Efficiency factor (0-1) for real-world performance (default: 0.3)
            
        Returns:
            Estimated latency per token in seconds
        """
        validate_positive_integer(flops_per_token, "FLOPs per token")
        validate_positive_number(gpu_tflops, "GPU TFLOPS")
        validate_positive_number(efficiency_factor, "Efficiency factor")
        
        if efficiency_factor > 1.0:
            raise ValueError(f"Efficiency factor must be <= 1.0, got {efficiency_factor}")
        
        # Convert GPU TFLOPS to FLOPS
        gpu_flops = gpu_tflops * (10 ** 12)
        
        # Calculate theoretical latency
        theoretical_latency = flops_per_token / gpu_flops
        
        # Apply efficiency factor for realistic estimate
        realistic_latency = theoretical_latency / efficiency_factor
        
        if self._history_callback:
            self._history_callback(
                f"Token_Generation_Latency(flops_per_token={flops_per_token}, "
                f"gpu_tflops={gpu_tflops}, efficiency_factor={efficiency_factor}) = "
                f"{realistic_latency:.4f} seconds/token"
            )
        
        return realistic_latency
    
    # Aliases for backward compatibility
    estimate_token_generation_latency = calculate_token_latency
    
    def calculate_completion_latency(self,
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
        validate_positive_integer(prompt_length, "Prompt length")
        validate_positive_integer(output_length, "Output length")
        validate_positive_integer(flops_prefill, "Prefill FLOPs")
        validate_positive_integer(flops_per_token, "FLOPs per token")
        validate_positive_number(gpu_tflops, "GPU TFLOPS")
        validate_positive_number(efficiency_factor, "Efficiency factor")
        
        # Calculate prefill latency
        prefill_latency = self.calculate_prefill_latency(
            flops_prefill, gpu_tflops, efficiency_factor
        )
        
        # Calculate per-token generation latency
        token_latency = self.calculate_token_latency(
            flops_per_token, gpu_tflops, efficiency_factor
        )
        
        # Calculate total generation latency
        generation_latency = token_latency * output_length
        
        # Calculate total latency
        total_latency = prefill_latency + generation_latency
        
        # Calculate tokens per second during generation
        tokens_per_second = 1.0 / token_latency
        
        result = {
            "prefill_latency": prefill_latency,
            "generation_latency": generation_latency,
            "total_latency": total_latency,
            "tokens_per_second": tokens_per_second,
            "time_to_first_token": prefill_latency
        }
        
        if self._history_callback:
            self._history_callback(
                f"Completion_Latency(prompt_length={prompt_length}, output_length={output_length}, "
                f"gpu_tflops={gpu_tflops}, efficiency_factor={efficiency_factor}) = "
                f"total: {total_latency:.2f}s, prefill: {prefill_latency:.2f}s, "
                f"generation: {generation_latency:.2f}s, tokens/sec: {tokens_per_second:.2f}"
            )
        
        return result
    
    # Alias for backward compatibility
    estimate_completion_latency = calculate_completion_latency
