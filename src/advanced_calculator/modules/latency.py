from typing import Dict, Any
from src.advanced_calculator.modules.utils import (
    validate_positive_integer,
    validate_positive_number,
)
import math  # Added for ceiling function

# Mapping precision strings to bytes per parameter
BYTES_PER_PARAM = {
    "fp32": 4,
    "tf32": 4,  # Technically uses fp32 storage
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
    "fp8": 1,  # E4M3 or E5M2 usually
    "int4": 0.5,  # 4-bit precision means half a byte per parameter
}


class LatencyCalculator:
    """
    Calculator for estimating latency of LLM operations, considering compute and memory.
    """

    def __init__(self, history_callback=None):
        """
        Initialize latency calculator

        Args:
            history_callback: Optional callback function to log calculation history
        """
        self._history_callback = history_callback

    def _get_bytes_per_param(self, precision: str) -> float:
        """Helper to get bytes per parameter based on precision string."""
        precision_lower = precision.lower()
        bytes_val = BYTES_PER_PARAM.get(precision_lower)
        if bytes_val is None:
            raise ValueError(
                f"Unsupported precision for latency calculation: {precision}"
            )
        return bytes_val

    def calculate_prefill_latency(
        self, flops_prefill: int, gpu_tflops: float, efficiency_factor: float = 0.3
    ) -> float:
        """
        Estimate prefill phase latency in seconds (currently compute-bound focused).

        TODO: Incorporate memory bandwidth effects for prefill (KV cache load, etc.)

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

        if efficiency_factor <= 0 or efficiency_factor > 1.0:
            raise ValueError(
                f"Efficiency factor must be between 0 (exclusive) and 1.0 (inclusive), got {efficiency_factor}"
            )

        # Convert GPU TFLOPS to FLOPS
        gpu_flops = gpu_tflops * (10**12)

        # Calculate theoretical latency
        theoretical_latency = (
            flops_prefill / gpu_flops if gpu_flops > 0 else float("inf")
        )

        # Apply efficiency factor for realistic estimate
        realistic_latency = theoretical_latency / efficiency_factor

        if self._history_callback:
            self._history_callback(
                f"Prefill_Latency(flops_prefill={flops_prefill}, "
                f"gpu_tflops={gpu_tflops}, efficiency_factor={efficiency_factor}) = "
                f"{realistic_latency:.4f} seconds (Compute-based)"
            )

        return realistic_latency

    # Alias for backward compatibility
    estimate_prefill_latency = calculate_prefill_latency

    def calculate_token_latency(
        self,
        flops_per_token: int,
        gpu_tflops: float,
        model_parameters_billion: float,
        gpu_bandwidth_gb_per_sec: float,
        precision: str,
        efficiency_factor: float = 0.3,
    ) -> float:
        """
        Estimate per-token generation latency in seconds, considering compute and memory bandwidth.
        The final latency is determined by the bottleneck (maximum) of compute and memory latency.

        Args:
            flops_per_token: FLOPs required to generate a single token.
            gpu_tflops: GPU throughput in TFLOPS (e.g., bf16).
            model_parameters_billion: Model size in billions of parameters.
            gpu_bandwidth_gb_per_sec: GPU memory bandwidth in GB/s.
            precision: The precision used (e.g., 'bf16', 'fp16', 'int8'). Determines bytes per parameter for memory calculation.
            efficiency_factor: Efficiency factor (0-1) applied to both compute and memory (default: 0.3).

        Returns:
            Estimated latency per token in seconds (bottleneck of compute vs. memory).
        """
        validate_positive_integer(flops_per_token, "FLOPs per token")
        validate_positive_number(gpu_tflops, "GPU TFLOPS")
        validate_positive_number(
            model_parameters_billion, "Model Parameters (Billions)"
        )
        validate_positive_number(gpu_bandwidth_gb_per_sec, "GPU Bandwidth (GB/s)")
        validate_positive_number(efficiency_factor, "Efficiency factor")

        if efficiency_factor <= 0 or efficiency_factor > 1.0:
            raise ValueError(
                f"Efficiency factor must be between 0 (exclusive) and 1.0 (inclusive), got {efficiency_factor}"
            )

        # --- Compute Latency ---
        gpu_flops = gpu_tflops * (10**12)
        theoretical_compute_latency = (
            flops_per_token / gpu_flops if gpu_flops > 0 else float("inf")
        )
        realistic_compute_latency = theoretical_compute_latency / efficiency_factor

        # --- Memory Latency ---
        bytes_per_param = self._get_bytes_per_param(precision)
        total_model_bytes = model_parameters_billion * (10**9) * bytes_per_param

        # Effective bandwidth considers efficiency
        effective_bandwidth_bytes_per_sec = (
            gpu_bandwidth_gb_per_sec * (10**9) * efficiency_factor
        )

        # Time = Bytes / (Bytes/Second)
        # Assuming all parameters need to be accessed from memory per token (common for batch size 1 decode)
        realistic_memory_latency = (
            total_model_bytes / effective_bandwidth_bytes_per_sec
            if effective_bandwidth_bytes_per_sec > 0
            else float("inf")
        )

        # --- Determine Bottleneck ---
        # The actual latency is limited by the slower of the two (compute or memory)
        realistic_latency = max(realistic_compute_latency, realistic_memory_latency)

        if self._history_callback:
            bottleneck = (
                "Memory"
                if realistic_memory_latency >= realistic_compute_latency
                else "Compute"
            )
            self._history_callback(
                f"Token_Generation_Latency(flops={flops_per_token}, tflops={gpu_tflops}, "
                f"params_b={model_parameters_billion}, bw_gbps={gpu_bandwidth_gb_per_sec}, "
                f"prec={precision}, eff={efficiency_factor}) = "
                f"{realistic_latency:.6f} s/token "
                f"(Compute: {realistic_compute_latency:.6f}s, Memory: {realistic_memory_latency:.6f}s, Bottleneck: {bottleneck})"
            )

        return realistic_latency

    # Aliases for backward compatibility
    estimate_token_generation_latency = calculate_token_latency

    def calculate_completion_latency(
        self,
        prompt_length: int,
        output_length: int,
        flops_prefill: int,
        flops_per_token: int,
        gpu_tflops: float,
        model_parameters_billion: float,
        gpu_bandwidth_gb_per_sec: float,
        precision: str,
        efficiency_factor: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Estimate end-to-end completion latency including prefill and token generation.
        Uses memory-aware calculation for token generation latency.

        Args:
            prompt_length: Length of input prompt in tokens.
            output_length: Expected output length in tokens.
            flops_prefill: Total FLOPs for prefill phase.
            flops_per_token: FLOPs per token for generation phase.
            gpu_tflops: GPU throughput in TFLOPS.
            model_parameters_billion: Model size in billions of parameters (used for token memory latency).
            gpu_bandwidth_gb_per_sec: GPU memory bandwidth in GB/s (used for token memory latency).
            precision: Precision used for calculation (e.g., 'bf16') (used for token memory latency).
            efficiency_factor: Efficiency factor for calculations.

        Returns:
            Dictionary with latency metrics (prefill_latency, generation_latency, total_latency, tokens_per_second, time_to_first_token).
        """
        validate_positive_integer(prompt_length, "Prompt length")
        validate_positive_integer(output_length, "Output length")
        validate_positive_integer(flops_prefill, "Prefill FLOPs")
        validate_positive_integer(flops_per_token, "FLOPs per token")
        validate_positive_number(gpu_tflops, "GPU TFLOPS")
        validate_positive_number(
            model_parameters_billion, "Model Parameters (Billions)"
        )
        validate_positive_number(gpu_bandwidth_gb_per_sec, "GPU Bandwidth (GB/s)")
        validate_positive_number(efficiency_factor, "Efficiency factor")

        # Calculate prefill latency (still compute-based for now)
        prefill_latency = self.calculate_prefill_latency(
            flops_prefill, gpu_tflops, efficiency_factor
        )

        # Calculate per-token generation latency (now includes memory bandwidth)
        token_latency = self.calculate_token_latency(
            flops_per_token,
            gpu_tflops,
            model_parameters_billion,
            gpu_bandwidth_gb_per_sec,
            precision,
            efficiency_factor,
        )

        # Calculate total generation latency
        generation_latency = token_latency * output_length

        # Calculate total latency
        total_latency = prefill_latency + generation_latency

        # Calculate tokens per second during generation
        # Handle potential division by zero if token_latency is somehow zero or infinite
        tokens_per_second = (
            (1.0 / token_latency)
            if token_latency > 0 and not math.isinf(token_latency)
            else 0.0
        )

        result = {
            "prefill_latency": prefill_latency,
            "generation_latency": generation_latency,
            "total_latency": total_latency,
            "tokens_per_second": tokens_per_second,
            "time_to_first_token": prefill_latency,  # Approximated as prefill latency
        }

        if self._history_callback:
            self._history_callback(
                f"Completion_Latency(prompt={prompt_length}, output={output_length}, "
                f"gpu_tflops={gpu_tflops}, eff={efficiency_factor}, "
                f"params_b={model_parameters_billion}, bw_gbps={gpu_bandwidth_gb_per_sec}, prec={precision}) = "
                f"total: {total_latency:.4f}s, prefill: {prefill_latency:.4f}s, "
                f"gen: {generation_latency:.4f}s, token_lat: {token_latency:.6f}s, tps: {tokens_per_second:.2f}"
            )

        return result

    # Alias for backward compatibility
    estimate_completion_latency = calculate_completion_latency
