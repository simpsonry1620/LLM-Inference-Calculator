"""
Predefined model architectures for LLM infrastructure calculations.

This module contains parameters for common LLM architectures to simplify resource estimation.
"""

from typing import Dict, Any, List, Optional, TypedDict

class ModelConfig(TypedDict, total=False):
    """Type definition for model configuration dictionary"""
    name: str
    family: str
    hidden_dimensions: int
    feedforward_dimensions: int
    num_layers: int
    vocab_size: int
    default_seq_length: int
    description: str
    parameter_count: float  # In billions


# Dictionary of predefined model architectures
KNOWN_MODELS: Dict[str, ModelConfig] = {
    # GPT-2 Models
    "gpt2-small": {
        "name": "GPT-2 Small",
        "family": "GPT-2",
        "hidden_dimensions": 768,
        "feedforward_dimensions": 3072,
        "num_layers": 12,
        "vocab_size": 50257,
        "default_seq_length": 1024,
        "description": "Original GPT-2 small model (124M parameters)",
        "parameter_count": 0.124
    },
    "gpt2-medium": {
        "name": "GPT-2 Medium",
        "family": "GPT-2",
        "hidden_dimensions": 1024,
        "feedforward_dimensions": 4096,
        "num_layers": 24,
        "vocab_size": 50257,
        "default_seq_length": 1024,
        "description": "Original GPT-2 medium model (355M parameters)",
        "parameter_count": 0.355
    },
    "gpt2-large": {
        "name": "GPT-2 Large",
        "family": "GPT-2",
        "hidden_dimensions": 1280,
        "feedforward_dimensions": 5120,
        "num_layers": 36,
        "vocab_size": 50257,
        "default_seq_length": 1024,
        "description": "Original GPT-2 large model (774M parameters)",
        "parameter_count": 0.774
    },
    "gpt2-xl": {
        "name": "GPT-2 XL",
        "family": "GPT-2",
        "hidden_dimensions": 1600,
        "feedforward_dimensions": 6400,
        "num_layers": 48,
        "vocab_size": 50257,
        "default_seq_length": 1024,
        "description": "Original GPT-2 XL model (1.5B parameters)",
        "parameter_count": 1.5
    },
    
    # Llama 2 Models
    "llama2-7b": {
        "name": "Llama 2 7B",
        "family": "Llama 2",
        "hidden_dimensions": 4096,
        "feedforward_dimensions": 11008,
        "num_layers": 32,
        "vocab_size": 32000,
        "default_seq_length": 4096,
        "description": "Meta's Llama 2 7B model",
        "parameter_count": 7.0
    },
    "llama2-13b": {
        "name": "Llama 2 13B",
        "family": "Llama 2",
        "hidden_dimensions": 5120,
        "feedforward_dimensions": 13824,
        "num_layers": 40,
        "vocab_size": 32000,
        "default_seq_length": 4096,
        "description": "Meta's Llama 2 13B model",
        "parameter_count": 13.0
    },
    "llama2-70b": {
        "name": "Llama 2 70B",
        "family": "Llama 2",
        "hidden_dimensions": 8192,
        "feedforward_dimensions": 28672,
        "num_layers": 80,
        "vocab_size": 32000,
        "default_seq_length": 4096,
        "description": "Meta's Llama 2 70B model",
        "parameter_count": 70.6
    },
    
    # Llama 3 Models
    "llama3-8b": {
        "name": "Llama 3 8B",
        "family": "Llama 3",
        "hidden_dimensions": 4096,
        "feedforward_dimensions": 14336,
        "num_layers": 32,
        "vocab_size": 128000,
        "default_seq_length": 8192,
        "description": "Meta's Llama 3 8B model",
        "parameter_count": 8.0
    },
    "llama3-70b": {
        "name": "Llama 3 70B",
        "family": "Llama 3",
        "hidden_dimensions": 8192,
        "feedforward_dimensions": 28672,
        "num_layers": 80,
        "vocab_size": 128000,
        "default_seq_length": 8192,
        "description": "Meta's Llama 3 70B model",
        "parameter_count": 70.6
    },
    
    # Llama 3.1 Models (using Llama 3 architecture but with 128k context)
    "llama3.1-8b": {
        "name": "Llama 3.1 8B",
        "family": "Llama 3.1",
        "hidden_dimensions": 4096,
        "feedforward_dimensions": 14336,
        "num_layers": 32,
        "vocab_size": 128256, # Llama 3 vocab size
        "default_seq_length": 131072, # 128k context window
        "description": "Meta's Llama 3.1 8B model (Llama 3 arch, 128k context)",
        "parameter_count": 8.0 # Assuming same params as Llama 3 8B
    },
    "llama3.1-70b": {
        "name": "Llama 3.1 70B",
        "family": "Llama 3.1",
        "hidden_dimensions": 8192,
        "feedforward_dimensions": 28672,
        "num_layers": 80,
        "vocab_size": 128256, # Llama 3 vocab size
        "default_seq_length": 131072, # 128k context window
        "description": "Meta's Llama 3.1 70B model (Llama 3 arch, 128k context)",
        "parameter_count": 70.6 # Assuming same params as Llama 3 70B
    },
    
    # Mistral Models
    "mistral-7b": {
        "name": "Mistral 7B",
        "family": "Mistral",
        "hidden_dimensions": 4096,
        "feedforward_dimensions": 14336,
        "num_layers": 32,
        "vocab_size": 32000,
        "default_seq_length": 8192,
        "description": "Mistral AI's 7B model with Sliding Window Attention",
        "parameter_count": 7.0
    },
    "mixtral-8x7b": {
        "name": "Mixtral 8x7B",
        "family": "Mistral",
        "hidden_dimensions": 4096,
        "feedforward_dimensions": 14336,
        "num_layers": 32,
        "vocab_size": 32000, 
        "default_seq_length": 32768,
        "description": "Mistral AI's 8x7B Mixture of Experts model (45B parameters active when routing)",
        "parameter_count": 45.0
    },
    
    # Phi Models
    "phi-1.5": {
        "name": "Phi-1.5",
        "family": "Phi",
        "hidden_dimensions": 2048,
        "feedforward_dimensions": 8192,
        "num_layers": 24,
        "vocab_size": 50257,
        "default_seq_length": 2048,
        "description": "Microsoft's Phi-1.5 small language model (1.3B parameters)",
        "parameter_count": 1.3
    },
    "phi-2": {
        "name": "Phi-2",
        "family": "Phi",
        "hidden_dimensions": 2560,
        "feedforward_dimensions": 10240, 
        "num_layers": 32,
        "vocab_size": 50257,
        "default_seq_length": 2048,
        "description": "Microsoft's Phi-2 small language model (2.7B parameters)",
        "parameter_count": 2.7
    },
    "phi-3-mini": {
        "name": "Phi-3 Mini",
        "family": "Phi",
        "hidden_dimensions": 3072,
        "feedforward_dimensions": 12288,
        "num_layers": 32,
        "vocab_size": 100000,
        "default_seq_length": 8192,
        "description": "Microsoft's Phi-3 Mini model (3.8B parameters)",
        "parameter_count": 3.8
    },
    "phi-3-small": {
        "name": "Phi-3 Small",
        "family": "Phi",
        "hidden_dimensions": 4096,
        "feedforward_dimensions": 16384,
        "num_layers": 32,
        "vocab_size": 100000,
        "default_seq_length": 8192,
        "description": "Microsoft's Phi-3 Small model (7B parameters)",
        "parameter_count": 7.0
    },
    "phi-3-medium": {
        "name": "Phi-3 Medium",
        "family": "Phi",
        "hidden_dimensions": 5120,
        "feedforward_dimensions": 20480,
        "num_layers": 36,
        "vocab_size": 100000,
        "default_seq_length": 8192,
        "description": "Microsoft's Phi-3 Medium model (14B parameters)",
        "parameter_count": 14.0
    },
    
    # Claude Models (estimated/approximate architecture)
    "claude-3-sonnet": {
        "name": "Claude 3 Sonnet",
        "family": "Claude",
        "hidden_dimensions": 8192,
        "feedforward_dimensions": 32768,
        "num_layers": 40,
        "vocab_size": 100000,
        "default_seq_length": 200000,
        "description": "Anthropic's Claude 3 Sonnet model (approximately 35B parameters)",
        "parameter_count": 35.0
    },
    "claude-3-opus": {
        "name": "Claude 3 Opus",
        "family": "Claude",
        "hidden_dimensions": 12288,
        "feedforward_dimensions": 49152,
        "num_layers": 60,
        "vocab_size": 100000,
        "default_seq_length": 200000,
        "description": "Anthropic's Claude 3 Opus model (approximately 115B parameters)",
        "parameter_count": 115.0
    },
    
    # Gemma Models
    "gemma-2b": {
        "name": "Gemma 2B",
        "family": "Gemma",
        "hidden_dimensions": 2048,
        "feedforward_dimensions": 16384,
        "num_layers": 18,
        "vocab_size": 256000,
        "default_seq_length": 8192,
        "description": "Google's Gemma 2B model",
        "parameter_count": 2.0
    },
    "gemma-7b": {
        "name": "Gemma 7B",
        "family": "Gemma",
        "hidden_dimensions": 3072,
        "feedforward_dimensions": 24576,
        "num_layers": 28,
        "vocab_size": 256000,
        "default_seq_length": 8192,
        "description": "Google's Gemma 7B model",
        "parameter_count": 7.0
    }
}


def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """
    Get configuration for a specific model by name.
    
    Args:
        model_name: Name of the model to retrieve
        
    Returns:
        Model configuration dictionary or None if model not found
    """
    return KNOWN_MODELS.get(model_name.lower())


def get_model_families() -> List[str]:
    """
    Get the list of available model families.
    
    Returns:
        List of unique model families
    """
    return sorted(list(set(model["family"] for model in KNOWN_MODELS.values())))


def get_models_by_family(family: str) -> List[ModelConfig]:
    """
    Get all models belonging to a specific family.
    
    Args:
        family: Name of the model family
        
    Returns:
        List of model configurations in the specified family
    """
    return [model for model in KNOWN_MODELS.values() if model["family"].lower() == family.lower()]


def list_all_models() -> List[str]:
    """
    List all available model names.
    
    Returns:
        List of model names
    """
    return sorted(KNOWN_MODELS.keys()) 