from typing import Literal, Dict, Any
from src.advanced_calculator.modules.utils import validate_positive_integer, validate_positive_number

class VRAMCalculator:
    """
    Calculator for estimating VRAM requirements for LLMs.
    """
    def __init__(self, history_callback=None):
        """
        Initialize VRAM calculator
        
        Args:
            history_callback: Optional callback function to log calculation history
        """
        self._history_callback = history_callback

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
        validate_positive_integer(hidden_dimensions, "Hidden dimensions")
        validate_positive_integer(feedforward_dimensions, "Feedforward dimensions")
        validate_positive_integer(num_layers, "Number of layers")
        validate_positive_integer(vocab_size, "Vocabulary size")
        validate_positive_number(overhead_factor, "Overhead factor")
        
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
        
        if self._history_callback:
            self._history_callback(
                f"Model_VRAM(hidden_dim={hidden_dimensions}, ff_dim={feedforward_dimensions}, "
                f"num_layers={num_layers}, vocab_size={vocab_size}, precision={precision}, "
                f"overhead_factor={overhead_factor}) = {gb_total_with_overhead:.2f} GB"
            )
        
        return gb_total_with_overhead
    
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
        validate_positive_integer(batch_size, "Batch size")
        validate_positive_integer(sequence_length, "Sequence length")
        validate_positive_integer(hidden_dimensions, "Hidden dimensions")
        validate_positive_integer(num_layers, "Number of layers")
        validate_positive_number(overhead_factor, "Overhead factor")
        
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
        
        if self._history_callback:
            self._history_callback(
                f"KV_Cache_VRAM(batch_size={batch_size}, sequence_length={sequence_length}, "
                f"hidden_dimensions={hidden_dimensions}, num_layers={num_layers}, "
                f"precision={precision}, overhead_factor={overhead_factor}) = {kv_cache_gb:.2f} GB"
            )
        
        return kv_cache_gb
        
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
        validate_positive_integer(batch_size, "Batch size")
        validate_positive_integer(sequence_length, "Sequence length")
        validate_positive_integer(hidden_dimensions, "Hidden dimensions")
        validate_positive_integer(num_layers, "Number of layers")
        validate_positive_number(overhead_factor, "Overhead factor")
        
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
        
        if self._history_callback:
            self._history_callback(
                f"Activations_VRAM(batch_size={batch_size}, sequence_length={sequence_length}, "
                f"hidden_dimensions={hidden_dimensions}, num_layers={num_layers}, "
                f"precision={precision}, overhead_factor={overhead_factor}) = {gb_total:.2f} GB"
            )
        
        return gb_total

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
        
        if self._history_callback:
            self._history_callback(
                f"Total_VRAM(weights={weights_vram:.2f} GB, "
                f"kv_cache={kv_cache_vram:.2f} GB, "
                f"other_activations={other_activations_vram:.2f} GB, "
                f"system_overhead={system_overhead}) = {total_vram:.2f} GB"
            )
        
        return total_vram

    def determine_model_scaling(self,
                               gpu_vram_gb: float,
                               total_vram_required_gb: float,
                               model_params: Dict[str, Any],
                               num_layers: int,
                               hidden_dimensions: int,
                               communication_overhead: float = 0.1) -> Dict[str, Any]:
        """
        Determine if a model fits on the selected GPU, and if not, how to scale it across multiple GPUs.
        
        This method analyzes the model's memory requirements against available GPU VRAM and recommends
        appropriate parallelization strategies if needed. It considers:
        - Tensor Parallelism (TP): Splitting individual tensors across GPUs
        - Pipeline Parallelism (PP): Distributing model layers across GPUs
        - Both TP and PP combined for very large models
        
        Args:
            gpu_vram_gb: VRAM capacity of a single GPU in GB
            total_vram_required_gb: Total VRAM required for the model in GB
            model_params: Dictionary containing model parameters and configurations
            num_layers: Number of transformer layers in the model
            hidden_dimensions: Size of hidden dimensions in the model
            communication_overhead: Fractional overhead due to communication between GPUs (default: 0.1 or 10%)
            
        Returns:
            Dictionary containing:
            - fits_on_single_gpu: Boolean indicating if model fits on a single GPU
            - num_gpus_required: Number of GPUs required for parallel execution
            - recommended_strategy: Recommended parallelism strategy ('single', 'tp', 'pp', 'tp+pp')
            - tp_degree: Recommended tensor parallelism degree (if applicable)
            - pp_degree: Recommended pipeline parallelism degree (if applicable)
            - estimated_efficiency: Estimated computational efficiency with parallelization
            - vram_per_gpu: Estimated VRAM required per GPU after parallelization
            - communication_overhead_gb: Estimated communication overhead in GB
        """
        # Check if model fits on a single GPU
        fits_on_single_gpu = total_vram_required_gb <= gpu_vram_gb
        
        if fits_on_single_gpu:
            result = {
                "fits_on_single_gpu": True,
                "num_gpus_required": 1,
                "recommended_strategy": "single",
                "tp_degree": 1,
                "pp_degree": 1,
                "estimated_efficiency": 1.0,
                "vram_per_gpu": total_vram_required_gb,
                "communication_overhead_gb": 0.0
            }
            
            if self._history_callback:
                self._history_callback(
                    f"Model fits on a single GPU with {gpu_vram_gb:.2f} GB VRAM. "
                    f"Required VRAM: {total_vram_required_gb:.2f} GB."
                )
                
            return result
            
        # If it doesn't fit, determine parallelization strategy
        # Calculate how many GPUs would be needed for simple data parallelism (duplicating the model)
        num_gpus_naive = max(2, int(total_vram_required_gb / gpu_vram_gb) + 1)
        
        # Determine if Tensor Parallelism is suitable based on hidden dimensions
        # For TP, hidden dimensions should be divisible by TP degree for optimal performance
        max_tp_degree = self._find_max_divisor(hidden_dimensions, upper_limit=8)
        
        # Determine if Pipeline Parallelism is suitable based on number of layers
        # For PP, number of layers should be divisible by PP degree for balanced pipeline stages
        max_pp_degree = self._find_max_divisor(num_layers, upper_limit=8)
        
        # Start with the smallest parallelization degree needed
        if num_gpus_naive <= max_tp_degree:
            tp_degree = num_gpus_naive
            pp_degree = 1
            strategy = "tp"
        elif num_gpus_naive <= max_pp_degree:
            tp_degree = 1
            pp_degree = num_gpus_naive
            strategy = "pp"
        else:
            # Need both TP and PP - find optimal combination
            tp_degree, pp_degree = self._find_optimal_tp_pp(
                num_gpus_naive, max_tp_degree, max_pp_degree, hidden_dimensions, num_layers
            )
            strategy = "tp+pp"
        
        # Calculate communication overhead based on parallelization strategy
        communication_overhead_gb = self._calculate_communication_overhead(
            strategy, tp_degree, pp_degree, hidden_dimensions, num_layers, 
            total_vram_required_gb, communication_overhead
        )
        
        # Calculate VRAM per GPU with parallelization
        vram_per_gpu = self._calculate_vram_per_gpu(
            total_vram_required_gb, tp_degree, pp_degree, strategy, communication_overhead_gb
        )
        
        # Calculate efficiency factor - lower with more GPUs due to communication overhead
        efficiency_factor = 1.0 - (0.05 * (tp_degree + pp_degree - 2))
        efficiency_factor = max(0.5, efficiency_factor)  # Ensure minimum 50% efficiency
        
        num_gpus_required = tp_degree * pp_degree
        
        result = {
            "fits_on_single_gpu": False,
            "num_gpus_required": num_gpus_required,
            "recommended_strategy": strategy,
            "tp_degree": tp_degree,
            "pp_degree": pp_degree,
            "estimated_efficiency": efficiency_factor,
            "vram_per_gpu": vram_per_gpu,
            "communication_overhead_gb": communication_overhead_gb
        }
        
        if self._history_callback:
            self._history_callback(
                f"Model requires parallelization across {num_gpus_required} GPUs. "
                f"Strategy: {strategy.upper()} with TP={tp_degree}, PP={pp_degree}. "
                f"VRAM per GPU: {vram_per_gpu:.2f} GB, Efficiency: {efficiency_factor:.2f}"
            )
            
        return result
    
    def _find_max_divisor(self, value: int, upper_limit: int = 8) -> int:
        """Find the largest divisor of value that is less than or equal to upper_limit."""
        for i in range(upper_limit, 0, -1):
            if value % i == 0 and i <= upper_limit:
                return i
        return 1
    
    def _find_optimal_tp_pp(self, 
                          target_gpus: int, 
                          max_tp: int,
                          max_pp: int,
                          hidden_dim: int,
                          num_layers: int) -> tuple:
        """
        Find optimal combination of tensor and pipeline parallelism degrees.
        
        Args:
            target_gpus: Target number of GPUs
            max_tp: Maximum tensor parallelism degree
            max_pp: Maximum pipeline parallelism degree
            hidden_dim: Model hidden dimension
            num_layers: Number of layers
            
        Returns:
            Tuple of (tp_degree, pp_degree)
        """
        # Try to find factors of target_gpus that are <= max_tp and max_pp
        best_tp, best_pp = 1, target_gpus  # Default to pure pipeline parallelism
        
        # Find all valid combinations
        valid_combinations = []
        for tp in range(1, min(max_tp, target_gpus) + 1):
            if target_gpus % tp == 0:  # tp is a factor of target_gpus
                pp = target_gpus // tp
                if pp <= max_pp:
                    valid_combinations.append((tp, pp))
        
        if not valid_combinations:
            # No exact factors found, find the closest combination
            best_product = 0
            for tp in range(1, max_tp + 1):
                for pp in range(1, max_pp + 1):
                    product = tp * pp
                    if product >= target_gpus and (best_product == 0 or product < best_product):
                        best_product = product
                        best_tp, best_pp = tp, pp
            return best_tp, best_pp
            
        # Score each combination based on balancing TP and PP
        best_score = float('inf')
        for tp, pp in valid_combinations:
            # Prefer balanced TP and PP, with slight preference for TP for better performance
            score = abs(tp - pp) + 0.1 * (pp - tp) if pp > tp else abs(tp - pp)
            
            # Consider alignment with model architecture
            tp_alignment = 0 if hidden_dim % tp == 0 else (tp - (hidden_dim % tp)) / tp
            pp_alignment = 0 if num_layers % pp == 0 else (pp - (num_layers % pp)) / pp
            
            # Lower score is better
            total_score = score + tp_alignment + pp_alignment
            
            if total_score < best_score:
                best_score = total_score
                best_tp, best_pp = tp, pp
                
        return best_tp, best_pp
    
    def _calculate_communication_overhead(self,
                                        strategy: str,
                                        tp_degree: int,
                                        pp_degree: int,
                                        hidden_dim: int,
                                        num_layers: int,
                                        total_vram_gb: float,
                                        overhead_factor: float) -> float:
        """
        Estimate communication overhead for different parallelization strategies.
        
        Args:
            strategy: Parallelization strategy ('tp', 'pp', or 'tp+pp')
            tp_degree: Tensor parallelism degree
            pp_degree: Pipeline parallelism degree
            hidden_dim: Model hidden dimension
            num_layers: Number of layers
            total_vram_gb: Total VRAM required without parallelization
            overhead_factor: Base communication overhead factor
            
        Returns:
            Estimated communication overhead in GB
        """
        if strategy == "single":
            return 0.0
            
        base_overhead = total_vram_gb * overhead_factor
        
        if strategy == "tp":
            # TP overhead scales with tensor parallelism degree
            # Higher TP means more all-reduce operations across GPUs
            return base_overhead * (tp_degree - 1) / tp_degree
            
        elif strategy == "pp":
            # PP overhead is typically lower than TP but scales with pipeline stages
            # Only need to communicate activations between pipeline stages
            return base_overhead * 0.5 * (pp_degree - 1) / pp_degree
            
        elif strategy == "tp+pp":
            # Combined overhead - multiply factors for both directions
            tp_factor = (tp_degree - 1) / tp_degree
            pp_factor = 0.5 * (pp_degree - 1) / pp_degree
            return base_overhead * (tp_factor + pp_factor)
            
        return 0.0
    
    def _calculate_vram_per_gpu(self,
                              total_vram_gb: float,
                              tp_degree: int,
                              pp_degree: int,
                              strategy: str,
                              communication_overhead_gb: float) -> float:
        """
        Calculate VRAM required per GPU after parallelization.
        
        Args:
            total_vram_gb: Total VRAM required without parallelization
            tp_degree: Tensor parallelism degree
            pp_degree: Pipeline parallelism degree
            strategy: Parallelization strategy
            communication_overhead_gb: Communication overhead in GB
            
        Returns:
            VRAM required per GPU in GB
        """
        if strategy == "single":
            return total_vram_gb
            
        num_gpus = tp_degree * pp_degree
        
        # Base calculation - divide total VRAM by number of GPUs
        base_vram_per_gpu = total_vram_gb / num_gpus
        
        # Add communication overhead per GPU
        overhead_per_gpu = communication_overhead_gb / num_gpus
        
        return base_vram_per_gpu + overhead_per_gpu
