"""
Calculator modules for different computation aspects of LLMs.
"""

from src.advanced_calculator.modules.flops import FLOPsCalculator
from src.advanced_calculator.modules.vram import VRAMCalculator
from src.advanced_calculator.modules.throughput import ThroughputCalculator
from src.advanced_calculator.modules.latency import LatencyCalculator
from src.advanced_calculator.modules.utils import HistoryManager, validate_positive_integer, validate_positive_number
from src.advanced_calculator.modules.models import (
    get_model_config, 
    list_all_models, 
    get_model_families, 
    get_models_by_family,
    ModelConfig,
    KNOWN_MODELS
)
from src.advanced_calculator.modules.gpus import (
    get_gpu_config,
    list_all_gpus,
    get_gpu_families,
    get_gpu_generations,
    get_gpus_by_family,
    get_gpus_by_generation,
    get_gpus_by_min_vram,
    get_gpus_supporting_precision,
    get_recommended_gpu_for_model,
    GPUConfig,
    KNOWN_GPUS
)

__all__ = [
    "FLOPsCalculator",
    "VRAMCalculator", 
    "ThroughputCalculator",
    "LatencyCalculator",
    "HistoryManager",
    "validate_positive_integer",
    "validate_positive_number",
    "get_model_config",
    "list_all_models",
    "get_model_families",
    "get_models_by_family",
    "ModelConfig",
    "KNOWN_MODELS",
    "get_gpu_config",
    "list_all_gpus",
    "get_gpu_families",
    "get_gpu_generations",
    "get_gpus_by_family",
    "get_gpus_by_generation",
    "get_gpus_by_min_vram",
    "get_gpus_supporting_precision",
    "get_recommended_gpu_for_model",
    "GPUConfig",
    "KNOWN_GPUS"
]
