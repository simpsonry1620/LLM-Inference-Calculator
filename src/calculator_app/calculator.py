#!/usr/bin/env python3
from typing import List, Union, Optional, Any, Literal, Dict
import math

class AdvancedCalculator:
    """
    Advanced calculator for estimating computational requirements of large language models.
    
    This calculator provides methods to estimate:
    - FLOPs (Floating Point Operations) for various components of transformer models
    - VRAM requirements for model weights
    - (More features to be added)
    
    These calculations can be useful for planning infrastructure requirements,
    estimating training and inference costs, and understanding scaling properties
    of large language models.
    """
    def __init__(self) -> None:
        self.history: List[str] = []
    
    def calculate_flops_attention(self, batch_size: int, sequence_length: int, hidden_dimensions: int) -> int:
        """
        Calculate FLOPs for attention mechanism.
        
        Args:
            batch_size: Number of sequences processed in parallel
            sequence_length: Length of input sequences in tokens
            hidden_dimensions: Size of hidden dimensions in the model
            
        Returns:
            Estimated FLOPs for attention computation
            
        Formula:
            FLOPs_Attention = Batch_Size * Sequence_Length * 
                             (Hidden_Dimensions^2 + Sequence_Length * Hidden_Dimensions)
        """
        self._validate_positive_integer(batch_size, "Batch size")
        self._validate_positive_integer(sequence_length, "Sequence length")
        self._validate_positive_integer(hidden_dimensions, "Hidden dimensions")
        
        hidden_dim_squared = hidden_dimensions ** 2
        seq_len_times_hidden = sequence_length * hidden_dimensions
        result = batch_size * sequence_length * (hidden_dim_squared + seq_len_times_hidden)
        
        self.history.append(
            f"FLOPs_Attention(batch_size={batch_size}, sequence_length={sequence_length}, "
            f"hidden_dimensions={hidden_dimensions}) = {result}"
        )
        return result
    
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
            
        Formula:
            FLOPs_Feedforward = 2 * Batch_Size * Sequence_Length * 
                               Hidden_Dimensions * FeedForward_Dimensions
        """
        self._validate_positive_integer(batch_size, "Batch size")
        self._validate_positive_integer(sequence_length, "Sequence length")
        self._validate_positive_integer(hidden_dimensions, "Hidden dimensions")
        self._validate_positive_integer(feedforward_dimensions, "Feedforward dimensions")
        
        result = 2 * batch_size * sequence_length * hidden_dimensions * feedforward_dimensions
        
        self.history.append(
            f"FLOPs_Feedforward(batch_size={batch_size}, sequence_length={sequence_length}, "
            f"hidden_dimensions={hidden_dimensions}, feedforward_dimensions={feedforward_dimensions}) = {result}"
        )
        return result
    
    def calculate_flops_prefill(self, batch_size: int, sequence_length: int, 
                               hidden_dimensions: int, feedforward_dimensions: int, num_layers: int) -> int:
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
            
        Formula:
            FLOPs_Prefill = Num_Layers * (FLOPs_Attention + FLOPs_Feedforward)
        """
        self._validate_positive_integer(batch_size, "Batch size")
        self._validate_positive_integer(sequence_length, "Sequence length")
        self._validate_positive_integer(hidden_dimensions, "Hidden dimensions")
        self._validate_positive_integer(feedforward_dimensions, "Feedforward dimensions")
        self._validate_positive_integer(num_layers, "Number of layers")
        
        flops_attention = self.calculate_flops_attention(batch_size, sequence_length, hidden_dimensions)
        flops_feedforward = self.calculate_flops_feedforward(batch_size, sequence_length, hidden_dimensions, feedforward_dimensions)
        
        result = num_layers * (flops_attention + flops_feedforward)
        
        self.history.append(
            f"FLOPs_Prefill(num_layers={num_layers}, "
            f"FLOPs_Attention={flops_attention}, FLOPs_Feedforward={flops_feedforward}) = {result}"
        )
        return result
    
    def calculate_model_vram(self, 
                            hidden_dimensions: int, 
                            feedforward_dimensions: int, 
                            num_layers: int,
                            vocab_size: int,
                            precision: Literal["fp16", "fp32", "bf16"] = "fp16",
                            overhead_factor: float = 1.0) -> float:
        """
        Calculate VRAM required for model weights.
        
        This method estimates the amount of VRAM needed to store the model parameters
        based on the model architecture and precision.
        
        Args:
            hidden_dimensions: Size of hidden dimensions in the model
            feedforward_dimensions: Size of feedforward network dimensions
            num_layers: Number of transformer layers
            vocab_size: Size of vocabulary for embeddings
            precision: Data precision (fp16=2 bytes, bf16=2 bytes, fp32=4 bytes)
            overhead_factor: Multiplicative factor to account for runtime overhead (default: 1.0)
            
        Returns:
            VRAM size in gigabytes (GB)
        """
        self._validate_positive_integer(hidden_dimensions, "Hidden dimensions")
        self._validate_positive_integer(feedforward_dimensions, "Feedforward dimensions")
        self._validate_positive_integer(num_layers, "Number of layers")
        self._validate_positive_integer(vocab_size, "Vocabulary size")
        self._validate_positive_number(overhead_factor, "Overhead factor")
        
        # Set bytes per parameter based on precision
        bytes_per_param = 4  # default for fp32
        if precision.lower() in ["fp16", "bf16"]:
            bytes_per_param = 2
            
        # Calculate parameters for attention layers
        # Query, Key, Value projections + output projection: 4 * hidden_dim * hidden_dim
        attention_params = 4 * hidden_dimensions * hidden_dimensions
        
        # Calculate parameters for feedforward network
        # Two fully connected layers: hidden_dim * ff_dim + ff_dim * hidden_dim
        ff_params = 2 * hidden_dimensions * feedforward_dimensions
        
        # Layer normalization parameters
        # 2 layer norms per transformer block, each with 2 * hidden_dim parameters (gamma and beta)
        layer_norm_params = 4 * hidden_dimensions
        
        # Total parameters per layer
        params_per_layer = attention_params + ff_params + layer_norm_params
        
        # Total parameters in all layers
        total_layer_params = num_layers * params_per_layer
        
        # Embedding parameters (input token embeddings)
        embedding_params = vocab_size * hidden_dimensions
        
        # Final layer norm
        final_layer_norm = 2 * hidden_dimensions
        
        # Total parameters
        total_params = total_layer_params + embedding_params + final_layer_norm
        
        # Convert to gigabytes (GB)
        bytes_total = total_params * bytes_per_param
        gb_total = bytes_total / (1024 ** 3)
        
        # Apply overhead factor
        gb_total_with_overhead = gb_total * overhead_factor
        
        self.history.append(
            f"Model_VRAM(hidden_dim={hidden_dimensions}, ff_dim={feedforward_dimensions}, "
            f"num_layers={num_layers}, vocab_size={vocab_size}, precision={precision}, "
            f"overhead_factor={overhead_factor}) = {gb_total_with_overhead:.2f} GB"
        )
        
        return gb_total_with_overhead
    
    def calculate_activations_vram(self,
                                  batch_size: int,
                                  sequence_length: int,
                                  hidden_dimensions: int,
                                  num_layers: int,
                                  precision: Literal["fp16", "fp32", "bf16"] = "fp16",
                                  overhead_factor: float = 1.1) -> float:
        """
        Calculate peak VRAM required for activations during inference.
        
        This method estimates the amount of VRAM needed for activations, including:
        - KV cache (key-value cache for attention mechanism)
        - Temporary activations in each layer
        
        Args:
            batch_size: Number of sequences processed in parallel
            sequence_length: Length of input sequences in tokens
            hidden_dimensions: Size of hidden dimensions in the model
            num_layers: Number of transformer layers
            precision: Data precision (fp16=2 bytes, bf16=2 bytes, fp32=4 bytes)
            overhead_factor: Multiplicative factor to account for runtime overhead (default: 1.1)
            
        Returns:
            Peak activations VRAM size in gigabytes (GB)
        """
        self._validate_positive_integer(batch_size, "Batch size")
        self._validate_positive_integer(sequence_length, "Sequence length")
        self._validate_positive_integer(hidden_dimensions, "Hidden dimensions")
        self._validate_positive_integer(num_layers, "Number of layers")
        self._validate_positive_number(overhead_factor, "Overhead factor")
        
        # Set bytes per value based on precision
        bytes_per_value = 4  # default for fp32
        if precision.lower() in ["fp16", "bf16"]:
            bytes_per_value = 2
            
        # Get KV cache size from dedicated function
        # Note: We use overhead_factor=1.0 here as we'll apply the overhead to the total later
        kv_cache_size = self.calculate_kv_cache_vram(
            batch_size, sequence_length, hidden_dimensions, num_layers, precision, 1.0
        ) * (1024 ** 3)  # Convert back to bytes for calculation
        
        # Temporary activations per layer during forward pass
        # This includes attention outputs, feedforward intermediate activations, etc.
        # Conservative estimate: 4 * batch_size * sequence_length * hidden_dimensions bytes per layer
        temp_activations_size = 4 * batch_size * sequence_length * hidden_dimensions * bytes_per_value
        
        # Total activation memory without overhead
        total_bytes = kv_cache_size + temp_activations_size
        
        # Apply overhead factor
        total_bytes_with_overhead = total_bytes * overhead_factor
        
        # Convert to gigabytes
        gb_total = total_bytes_with_overhead / (1024 ** 3)
        
        self.history.append(
            f"Activations_VRAM(batch_size={batch_size}, sequence_length={sequence_length}, "
            f"hidden_dimensions={hidden_dimensions}, num_layers={num_layers}, "
            f"precision={precision}, overhead_factor={overhead_factor}) = {gb_total:.2f} GB"
        )
        
        return gb_total
    
    def calculate_kv_cache_vram(self,
                              batch_size: int,
                              sequence_length: int,
                              hidden_dimensions: int,
                              num_layers: int,
                              precision: Literal["fp16", "fp32", "bf16"] = "fp16",
                              overhead_factor: float = 1.0) -> float:
        """
        Calculate VRAM required for KV (Key-Value) cache during inference.
        
        The KV cache stores the key and value tensors for previously processed tokens,
        which is essential for autoregressive generation to avoid recomputation.
        
        Args:
            batch_size: Number of sequences processed in parallel
            sequence_length: Length of input sequences in tokens
            hidden_dimensions: Size of hidden dimensions in the model
            num_layers: Number of transformer layers
            precision: Data precision (fp16=2 bytes, bf16=2 bytes, fp32=4 bytes)
            overhead_factor: Multiplicative factor to account for memory alignment and padding (default: 1.0)
            
        Returns:
            KV cache VRAM size in gigabytes (GB)
        """
        self._validate_positive_integer(batch_size, "Batch size")
        self._validate_positive_integer(sequence_length, "Sequence length")
        self._validate_positive_integer(hidden_dimensions, "Hidden dimensions")
        self._validate_positive_integer(num_layers, "Number of layers")
        self._validate_positive_number(overhead_factor, "Overhead factor")
        
        # Set bytes per value based on precision
        bytes_per_value = 4  # default for fp32
        if precision.lower() in ["fp16", "bf16"]:
            bytes_per_value = 2
            
        # KV cache is the largest component of activation memory
        # For each layer, we store K and V tensors of shape [batch_size, sequence_length, hidden_dimensions]
        kv_cache_bytes = 2 * batch_size * sequence_length * hidden_dimensions * num_layers * bytes_per_value
        
        # Apply overhead factor for memory alignment and padding
        kv_cache_bytes_with_overhead = kv_cache_bytes * overhead_factor
        
        # Convert to gigabytes
        kv_cache_gb = kv_cache_bytes_with_overhead / (1024 ** 3)
        
        self.history.append(
            f"KV_Cache_VRAM(batch_size={batch_size}, sequence_length={sequence_length}, "
            f"hidden_dimensions={hidden_dimensions}, num_layers={num_layers}, "
            f"precision={precision}, overhead_factor={overhead_factor}) = {kv_cache_gb:.2f} GB"
        )
        
        return kv_cache_gb
    
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
        
        This method sums the VRAM required for:
        - Model weights
        - KV cache
        - Other activations
        - Plus a system overhead
        
        Args:
            batch_size: Number of sequences processed in parallel
            sequence_length: Length of input sequences in tokens
            hidden_dimensions: Size of hidden dimensions in the model
            feedforward_dimensions: Size of feedforward network dimensions
            num_layers: Number of transformer layers
            vocab_size: Size of vocabulary for embeddings
            precision: Data precision (fp16=2 bytes, bf16=2 bytes, fp32=4 bytes)
            weights_overhead: Overhead factor for model weights (default: 1.05)
            kv_cache_overhead: Overhead factor for KV cache (default: 1.05)
            activations_overhead: Overhead factor for other activations (default: 1.1)
            system_overhead: Overall system overhead factor applied to the sum (default: 1.05)
            
        Returns:
            Total VRAM size in gigabytes (GB)
        """
        # Calculate model weights VRAM
        weights_vram = self.calculate_model_vram(
            hidden_dimensions,
            feedforward_dimensions,
            num_layers,
            vocab_size,
            precision,
            weights_overhead
        )
        
        # Calculate KV cache VRAM
        kv_cache_vram = self.calculate_kv_cache_vram(
            batch_size,
            sequence_length,
            hidden_dimensions,
            num_layers,
            precision,
            kv_cache_overhead
        )
        
        # Calculate other activations VRAM with specific overhead
        # We first get the raw activations without overhead to avoid double-counting
        raw_activations_vram = self.calculate_activations_vram(
            batch_size,
            sequence_length,
            hidden_dimensions,
            num_layers,
            precision,
            1.0  # No overhead here, we'll apply it separately
        )
        
        raw_kv_cache_vram = self.calculate_kv_cache_vram(
            batch_size,
            sequence_length,
            hidden_dimensions,
            num_layers,
            precision,
            1.0  # No overhead here either
        )
        
        other_activations_raw = raw_activations_vram - raw_kv_cache_vram
        other_activations_vram = other_activations_raw * activations_overhead
        
        # Sum components
        components_sum = weights_vram + kv_cache_vram + other_activations_vram
        
        # Apply overall system overhead
        total_vram = components_sum * system_overhead
        
        self.history.append(
            f"Total_VRAM(weights={weights_vram:.2f} GB, "
            f"kv_cache={kv_cache_vram:.2f} GB, "
            f"other_activations={other_activations_vram:.2f} GB, "
            f"system_overhead={system_overhead}) = {total_vram:.2f} GB"
        )
        
        return total_vram
    
    def clear_history(self) -> None:
        """Clear calculation history"""
        self.history = []
    
    def get_history(self) -> List[str]:
        """Return calculation history"""
        return self.history
    
    def _validate_positive_integer(self, value: Any, param_name: str) -> None:
        """Validate that the input is a positive integer"""
        if not isinstance(value, int):
            raise TypeError(f"{param_name} must be an integer, got {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"{param_name} must be positive, got {value}")
    
    def _validate_positive_number(self, value: Any, param_name: str) -> None:
        """Validate that the input is a positive number"""
        if not isinstance(value, (int, float)):
            raise TypeError(f"{param_name} must be a number, got {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"{param_name} must be positive, got {value}")

    def determine_model_scaling(self,
                                gpu_vram_gb: float,
                                # Need more inputs to calculate VRAM per GPU accurately
                                batch_size: int,
                                sequence_length: int,
                                hidden_dimensions: int,
                                feedforward_dimensions: int,
                                num_layers: int,
                                vocab_size: int,
                                precision: Literal["fp16", "fp32", "bf16"] = "fp16",
                                # Overheads might be needed too, use defaults for now
                                weights_overhead: float = 1.05,
                                kv_cache_overhead: float = 1.05,
                                activations_overhead: float = 1.1,
                                system_overhead: float = 1.05
                               ) -> Dict[str, Any]:
        """
        Determines if a model fits on a single GPU and suggests a scaling strategy if not.

        Args:
            gpu_vram_gb: VRAM capacity of a single GPU in GB.
            batch_size: Batch size for inference.
            sequence_length: Sequence length.
            hidden_dimensions: Model hidden dimensions.
            feedforward_dimensions: Model feedforward dimensions.
            num_layers: Number of model layers.
            vocab_size: Model vocabulary size.
            precision: Model precision.
            weights_overhead: Overhead factor for weights VRAM.
            kv_cache_overhead: Overhead factor for KV cache VRAM.
            activations_overhead: Overhead factor for activations VRAM.
            system_overhead: Overall system overhead factor.


        Returns:
            Dictionary containing scaling analysis results:
            - fits_on_single_gpu: bool
            - num_gpus_required: int
            - recommended_strategy: str
            - tp_degree: int
            - pp_degree: int
            - estimated_efficiency: float (placeholder)
            - vram_per_gpu: float
            - communication_overhead_gb: float (placeholder)
        """
        self._validate_positive_number(gpu_vram_gb, "GPU VRAM")
        # Recalculate total VRAM needed using provided parameters
        total_vram_required_gb = self.calculate_total_vram(
             batch_size=batch_size,
             sequence_length=sequence_length,
             hidden_dimensions=hidden_dimensions,
             feedforward_dimensions=feedforward_dimensions,
             num_layers=num_layers,
             vocab_size=vocab_size,
             precision=precision,
             weights_overhead=weights_overhead,
             kv_cache_overhead=kv_cache_overhead,
             activations_overhead=activations_overhead,
             system_overhead=system_overhead
        )


        fits_on_single_gpu = total_vram_required_gb <= gpu_vram_gb
        num_gpus_required = 1
        tp_degree = 1
        pp_degree = 1
        vram_per_gpu_gb = total_vram_required_gb # Default if fits on one GPU
        recommended_strategy = "Single GPU"

        if not fits_on_single_gpu:
            # Basic heuristic: Prioritize TP up to 8, then add PP
            # Calculate raw minimum GPUs needed
            min_gpus_needed = math.ceil(total_vram_required_gb / gpu_vram_gb)

            # Determine TP degree
            tp_degree = min(min_gpus_needed, 8) # Max TP degree of 8 as a heuristic

            # Determine PP degree
            # How many GPUs are needed per TP group to hold the sharded model?
            # Need individual VRAM components for better PP calculation.
            # Simple approach: Distribute remaining requirement over PP stages.
            pp_degree = math.ceil(min_gpus_needed / tp_degree)

            num_gpus_required = tp_degree * pp_degree
            recommended_strategy = f"TP={tp_degree}, PP={pp_degree}"

            # Estimate VRAM per GPU (simplified)
            # More accurate calculation requires knowing how components are sharded.
            # Model weights sharded by TP, KV/Activations by PP (often)
            model_vram_gb = self.calculate_model_vram(hidden_dimensions, feedforward_dimensions, num_layers, vocab_size, precision, weights_overhead)
            kv_cache_gb = self.calculate_kv_cache_vram(max(1, batch_size // pp_degree), sequence_length, hidden_dimensions, num_layers, precision, kv_cache_overhead)
            activations_gb = self.calculate_activations_vram(max(1, batch_size // pp_degree), sequence_length, hidden_dimensions, num_layers, precision, activations_overhead)

            # VRAM per GPU depends on the specific TP/PP implementation details.
            # Simplistic estimate: Assume model VRAM splits by TP, others don't (or handled by PP batch split)
            # A better model would distribute KV/Activations by PP.
            vram_model_per_gpu = model_vram_gb / tp_degree
            # KV and activations are often calculated per PP stage batch size
            vram_per_gpu_gb = (vram_model_per_gpu + kv_cache_gb + activations_gb) * system_overhead


            # If the calculated VRAM per GPU is still too high, adjust strategy (e.g., increase PP)
            # This basic heuristic might need refinement for complex cases.
            if vram_per_gpu_gb > gpu_vram_gb and pp_degree < num_layers : # Check if adding PP is possible and needed
                 # Try increasing PP
                 pp_degree_new = math.ceil(total_vram_required_gb / (gpu_vram_gb * tp_degree)) # Estimate required PP stages
                 pp_degree = max(pp_degree, pp_degree_new) # Ensure we don't decrease pp_degree
                 num_gpus_required = tp_degree * pp_degree
                 recommended_strategy = f"TP={tp_degree}, PP={pp_degree} (adjusted)"

                 # Recalculate per-GPU VRAM with new PP degree
                 kv_cache_gb = self.calculate_kv_cache_vram(max(1, batch_size // pp_degree), sequence_length, hidden_dimensions, num_layers, precision, kv_cache_overhead)
                 activations_gb = self.calculate_activations_vram(max(1, batch_size // pp_degree), sequence_length, hidden_dimensions, num_layers, precision, activations_overhead)
                 vram_per_gpu_gb = (vram_model_per_gpu + kv_cache_gb + activations_gb) * system_overhead


        return {
            "fits_on_single_gpu": fits_on_single_gpu,
            "total_vram_required_gb": round(total_vram_required_gb, 2), # Add this for context
            "num_gpus_required": num_gpus_required,
            "recommended_strategy": recommended_strategy,
            "tp_degree": tp_degree,
            "pp_degree": pp_degree,
            "estimated_efficiency": 0.7,  # Placeholder
            "vram_per_gpu": round(vram_per_gpu_gb, 2),
            "communication_overhead_gb": 0.1 * (num_gpus_required -1) # Simple placeholder based on num GPUs
        }


if __name__ == "__main__":
    # Example usage with different model sizes
    calc = AdvancedCalculator()
    
    # ===== Test 1: Small model (similar to GPT-2) =====
    print("\n===== Small model (GPT-2 like) =====")
    batch = 32
    seq_len = 512
    hidden_dim = 768
    ff_dim = 3072  # 4x hidden_dim
    num_layers = 12
    vocab_size = 50257
    
    print(f"FLOPs Attention: {calc.calculate_flops_attention(batch, seq_len, hidden_dim):,}")
    print(f"FLOPs Feedforward: {calc.calculate_flops_feedforward(batch, seq_len, hidden_dim, ff_dim):,}")
    print(f"FLOPs Prefill: {calc.calculate_flops_prefill(batch, seq_len, hidden_dim, ff_dim, num_layers):,}")
    print(f"Model VRAM (FP16): {calc.calculate_model_vram(hidden_dim, ff_dim, num_layers, vocab_size):,.2f} GB")
    print(f"KV Cache VRAM (FP16): {calc.calculate_kv_cache_vram(batch, seq_len, hidden_dim, num_layers):,.2f} GB")
    print(f"Activations VRAM (FP16): {calc.calculate_activations_vram(batch, seq_len, hidden_dim, num_layers):,.2f} GB")
    print(f"Total VRAM: {calc.calculate_total_vram(batch, seq_len, hidden_dim, ff_dim, num_layers, vocab_size):,.2f} GB")
    
    # ===== Test 2: Llama 3.1 70B =====
    # Using publicly known/estimated parameters for Llama 3.1 70B
    print("\n===== Llama 3.1 70B =====")
    
    batch = 1  # Typical inference batch size
    seq_len = 8192  # Context length capability
    hidden_dim = 8192  # Estimated hidden dimension
    ff_dim = 28672  # Estimated feedforward dimension (3.5x hidden_dim)
    num_layers = 80  # Estimated number of layers
    vocab_size = 128000  # Estimated vocabulary size
    
    # Example with detailed overhead factors
    weights_oh = 1.05  # Extra overhead for model weights
    kv_cache_oh = 1.05  # Overhead for KV cache due to memory alignment
    activations_oh = 1.2  # Higher overhead for activations due to framework overhead
    system_oh = 1.1  # Overall system overhead for CUDA context, etc.
    
    print(f"FLOPs Attention: {calc.calculate_flops_attention(batch, seq_len, hidden_dim):,}")
    print(f"FLOPs Feedforward: {calc.calculate_flops_feedforward(batch, seq_len, hidden_dim, ff_dim):,}")
    print(f"FLOPs Prefill: {calc.calculate_flops_prefill(batch, seq_len, hidden_dim, ff_dim, num_layers):,}")
    print(f"Model VRAM (FP16, with overhead): {calc.calculate_model_vram(hidden_dim, ff_dim, num_layers, vocab_size, overhead_factor=weights_oh):,.2f} GB")
    print(f"KV Cache VRAM (FP16, with overhead): {calc.calculate_kv_cache_vram(batch, seq_len, hidden_dim, num_layers, overhead_factor=kv_cache_oh):,.2f} GB")
    print(f"Activations VRAM (FP16, with overhead): {calc.calculate_activations_vram(batch, seq_len, hidden_dim, num_layers, overhead_factor=activations_oh):,.2f} GB")
    print(f"Total VRAM (with all overheads): {calc.calculate_total_vram(batch, seq_len, hidden_dim, ff_dim, num_layers, vocab_size, weights_overhead=weights_oh, kv_cache_overhead=kv_cache_oh, activations_overhead=activations_oh, system_overhead=system_oh):,.2f} GB")
    
    # Calculate theoretical throughput assuming A100 GPU with 312 TFLOPS (FP16)
    a100_tflops = 312  # A100 theoretical peak performance in TFLOPS (FP16)
    flops_prefill = calc.calculate_flops_prefill(batch, seq_len, hidden_dim, ff_dim, num_layers)
    theoretical_tokens_per_second = (a100_tflops * 10**12) / (flops_prefill / seq_len)
    print(f"Theoretical tokens/second on A100 (at 100% efficiency): {theoretical_tokens_per_second:.2f}")
    print(f"Realistic tokens/second (at 30% efficiency): {theoretical_tokens_per_second * 0.3:.2f}")
    
    print("\n===== Calculation History =====")
    for entry in calc.get_history():
        print(entry) 