"""
LLM scaling calculator for infrastructure estimation.
"""
import numpy as np
import os
import json
import datetime


class LLMScalingCalculator:
    """
    Calculator for estimating LLM infrastructure requirements.
    """
    
    def __init__(self, gpu_memory_gb=80, gpu_flops=312e12, gpu_cost_per_hour=1.5):
        """
        Initialize the LLM scaling calculator.
        
        Args:
            gpu_memory_gb: Memory per GPU in GB (default: 80GB, A100)
            gpu_flops: Peak FP16 FLOPS per GPU (default: 312 TFLOPS for A100)
            gpu_cost_per_hour: Cost per GPU hour in USD (default: $1.5)
        """
        self.gpu_memory_gb = gpu_memory_gb
        self.gpu_flops = gpu_flops
        self.gpu_cost_per_hour = gpu_cost_per_hour
        
    def estimate_memory_requirements(self, model_size_params, batch_size, sequence_length, 
                                    activation_factor=5, optimizer_factor=12):
        """
        Estimate memory requirements for training.
        
        Args:
            model_size_params: Number of model parameters
            batch_size: Training batch size
            sequence_length: Sequence length for training
            activation_factor: Memory multiplier for activations (default: 5)
            optimizer_factor: Memory multiplier for optimizer states (default: 12 for Adam)
            
        Returns:
            Dictionary with memory requirements in GB
        """
        # Model parameters (FP16: 2 bytes per parameter)
        model_size_gb = model_size_params * 2 / (1024**3)
        
        # Optimizer states (typically 12x model size for Adam)
        optimizer_size_gb = model_size_gb * optimizer_factor
        
        # Activations
        tokens_per_batch = batch_size * sequence_length
        activation_size_gb = model_size_gb * activation_factor * tokens_per_batch / 1_000_000
        
        # Total memory
        total_memory_gb = model_size_gb + optimizer_size_gb + activation_size_gb
        
        # Number of GPUs needed
        gpus_needed = np.ceil(total_memory_gb / self.gpu_memory_gb)
        
        return {
            "model_size_gb": model_size_gb,
            "optimizer_size_gb": optimizer_size_gb,
            "activation_size_gb": activation_size_gb,
            "total_memory_gb": total_memory_gb,
            "gpus_needed": int(gpus_needed)
        }
    
    def estimate_compute_requirements(self, model_size_params, batch_size, 
                                     sequence_length, tokens_to_train=300e9):
        """
        Estimate compute requirements for training.
        
        Args:
            model_size_params: Number of model parameters
            batch_size: Training batch size
            sequence_length: Sequence length for training
            tokens_to_train: Total tokens to train on (default: 300B)
            
        Returns:
            Dictionary with compute requirements
        """
        # FLOPs per token (forward + backward pass)
        flops_per_token = 6 * model_size_params * sequence_length
        
        # Total FLOPs for full training run
        total_flops = flops_per_token * tokens_to_train
        
        # Convert to more readable units
        total_petaflops = total_flops / 1e15
        
        # GPU hours
        gpu_hours = total_flops / self.gpu_flops / 3600
        
        # Training time with multiple GPUs
        memory_reqs = self.estimate_memory_requirements(
            model_size_params, batch_size, sequence_length
        )
        gpus_needed = memory_reqs["gpus_needed"]
        
        # Assume 50% efficiency at scale
        efficiency = min(1.0, 2.0 / np.sqrt(gpus_needed))
        
        training_hours = gpu_hours / (gpus_needed * efficiency)
        training_days = training_hours / 24
        
        # Cost
        training_cost = gpu_hours * self.gpu_cost_per_hour
        
        return {
            "flops_per_token": flops_per_token,
            "total_flops": total_flops,
            "total_petaflops": total_petaflops,
            "gpu_hours": gpu_hours,
            "gpus_needed": gpus_needed,
            "training_hours": training_hours,
            "training_days": training_days,
            "training_cost_usd": training_cost,
            "scaling_efficiency": efficiency
        }
    
    def estimate_resources(self, model_size_params, batch_size, sequence_length, 
                          tokens_to_train=300e9):
        """
        Estimate all resources for LLM training.
        
        Args:
            model_size_params: Number of model parameters
            batch_size: Training batch size
            sequence_length: Sequence length for training
            tokens_to_train: Total tokens to train on (default: 300B)
            
        Returns:
            Dictionary with all resource estimations
        """
        memory_reqs = self.estimate_memory_requirements(
            model_size_params, batch_size, sequence_length
        )
        
        compute_reqs = self.estimate_compute_requirements(
            model_size_params, batch_size, sequence_length, tokens_to_train
        )
        
        results = {
            "memory_requirements": memory_reqs,
            "compute_requirements": compute_reqs,
            "summary": {
                "model_size_billions": model_size_params / 1e9,
                "gpus_needed": memory_reqs["gpus_needed"],
                "training_days": compute_reqs["training_days"],
                "training_cost_usd": compute_reqs["training_cost_usd"]
            }
        }
        
        # Log the calculation results to JSON file
        self._log_calculation_to_json(results, model_size_params, batch_size, sequence_length, tokens_to_train)
        
        return results

    def _log_calculation_to_json(self, results, model_size_params, batch_size, sequence_length, tokens_to_train):
        """
        Log calculation results to a JSON file in logs/calculations/ directory.
        
        Args:
            results: The calculation results dictionary
            model_size_params: Number of model parameters
            batch_size: Training batch size
            sequence_length: Sequence length for training
            tokens_to_train: Total tokens to train on
        """
        # Create logs/calculations directory if it doesn't exist
        log_dir = os.path.join("logs", "calculations")
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate timestamp for the filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a descriptive filename
        model_size_b = model_size_params / 1e9
        filename = f"llm_calc_{model_size_b:.1f}B_batch{batch_size}_seq{sequence_length}_{timestamp}.json"
        
        # Full file path
        file_path = os.path.join(log_dir, filename)
        
        # Add metadata to the results
        log_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "input_parameters": {
                "model_size_params": model_size_params,
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "tokens_to_train": tokens_to_train,
                "gpu_memory_gb": self.gpu_memory_gb,
                "gpu_flops": self.gpu_flops,
                "gpu_cost_per_hour": self.gpu_cost_per_hour
            },
            "results": results
        }
        
        # Write the results to the JSON file
        with open(file_path, "w") as f:
            json.dump(log_data, f, indent=2) 