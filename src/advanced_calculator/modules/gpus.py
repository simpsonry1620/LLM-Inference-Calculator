"""
Predefined GPU specifications for LLM infrastructure calculations.

This module contains specifications for common NVIDIA GPUs to simplify resource estimation.
"""

from typing import Dict, Any, List, Optional, TypedDict, Set, Literal

class GPUConfig(TypedDict, total=False):
    """Type definition for GPU configuration dictionary"""
    name: str
    family: str
    gen: str
    vram_gb: float
    bandwidth_gb_per_sec: float
    fp32_tflops: float
    fp16_tflops: float
    bf16_tflops: float
    int8_tflops: float
    int4_tflops: float
    tdp_watts: int
    tensor_cores: bool
    max_batch_size: int
    launch_year: int
    description: str
    pcie_gen: int
    interconnect_bandwidth_gb_per_sec: float
    compute_capability: str
    supported_precisions: List[str]


# Dictionary of predefined NVIDIA GPUs
KNOWN_GPUS: Dict[str, GPUConfig] = {
    # RTX 30 Series (Ampere)
    "rtx-3090": {
        "name": "NVIDIA RTX 3090",
        "family": "RTX",
        "gen": "Ampere",
        "vram_gb": 24.0,
        "bandwidth_gb_per_sec": 936.0,
        "fp32_tflops": 35.6,
        "fp16_tflops": 71.2,
        "bf16_tflops": 71.2,
        "int8_tflops": 142.0,
        "int4_tflops": 284.0,
        "tdp_watts": 350,
        "tensor_cores": True,
        "max_batch_size": 16,
        "launch_year": 2020,
        "description": "Consumer flagship GPU of the Ampere generation",
        "pcie_gen": 4,
        "interconnect_bandwidth_gb_per_sec": 32.0,
        "compute_capability": "8.6",
        "supported_precisions": ["fp32", "fp16", "bf16", "int8", "int4"]
    },
    "rtx-3080": {
        "name": "NVIDIA RTX 3080",
        "family": "RTX",
        "gen": "Ampere",
        "vram_gb": 10.0,
        "bandwidth_gb_per_sec": 760.0,
        "fp32_tflops": 29.8,
        "fp16_tflops": 59.6,
        "bf16_tflops": 59.6,
        "int8_tflops": 119.0,
        "int4_tflops": 238.0,
        "tdp_watts": 320,
        "tensor_cores": True,
        "max_batch_size": 8,
        "launch_year": 2020,
        "description": "High-end consumer GPU of the Ampere generation",
        "pcie_gen": 4, 
        "interconnect_bandwidth_gb_per_sec": 32.0,
        "compute_capability": "8.6",
        "supported_precisions": ["fp32", "fp16", "bf16", "int8", "int4"]
    },
    
    # RTX 40 Series (Ada Lovelace)
    "rtx-4090": {
        "name": "NVIDIA RTX 4090",
        "family": "RTX",
        "gen": "Ada Lovelace",
        "vram_gb": 24.0,
        "bandwidth_gb_per_sec": 1008.0,
        "fp32_tflops": 82.6,
        "fp16_tflops": 165.0,
        "bf16_tflops": 165.0,
        "int8_tflops": 330.0,
        "int4_tflops": 660.0,
        "tdp_watts": 450,
        "tensor_cores": True,
        "max_batch_size": 16,
        "launch_year": 2022,
        "description": "Flagship consumer GPU of the Ada Lovelace generation",
        "pcie_gen": 4,
        "interconnect_bandwidth_gb_per_sec": 32.0,
        "compute_capability": "8.9",
        "supported_precisions": ["fp32", "fp16", "bf16", "int8", "int4"]
    },
    "rtx-4080": {
        "name": "NVIDIA RTX 4080",
        "family": "RTX",
        "gen": "Ada Lovelace",
        "vram_gb": 16.0,
        "bandwidth_gb_per_sec": 717.0,
        "fp32_tflops": 48.7,
        "fp16_tflops": 97.4,
        "bf16_tflops": 97.4,
        "int8_tflops": 194.8,
        "int4_tflops": 389.6,
        "tdp_watts": 320,
        "tensor_cores": True,
        "max_batch_size": 12,
        "launch_year": 2022,
        "description": "High-end consumer GPU of the Ada Lovelace generation",
        "pcie_gen": 4,
        "interconnect_bandwidth_gb_per_sec": 32.0,
        "compute_capability": "8.9",
        "supported_precisions": ["fp32", "fp16", "bf16", "int8", "int4"]
    },
    
    # Datacenter GPUs - Ampere
    "a100-sxm4-40gb": {
        "name": "NVIDIA A100 SXM4 40GB",
        "family": "Datacenter",
        "gen": "Ampere",
        "vram_gb": 40.0,
        "bandwidth_gb_per_sec": 1555.0,
        "fp32_tflops": 19.5,
        "fp16_tflops": 312.0,
        "bf16_tflops": 312.0,
        "int8_tflops": 624.0,
        "int4_tflops": 1248.0,
        "tdp_watts": 400,
        "tensor_cores": True,
        "max_batch_size": 32,
        "launch_year": 2020,
        "description": "Datacenter GPU with SXM4 form factor",
        "pcie_gen": 4,
        "interconnect_bandwidth_gb_per_sec": 600.0,
        "compute_capability": "8.0",
        "supported_precisions": ["fp32", "tf32", "fp16", "bf16", "int8", "int4"]
    },
    "a100-pcie-40gb": {
        "name": "NVIDIA A100 PCIe 40GB",
        "family": "Datacenter",
        "gen": "Ampere",
        "vram_gb": 40.0,
        "bandwidth_gb_per_sec": 1555.0,
        "fp32_tflops": 19.5,
        "fp16_tflops": 312.0,
        "bf16_tflops": 312.0,
        "int8_tflops": 624.0,
        "int4_tflops": 1248.0,
        "tdp_watts": 250,
        "tensor_cores": True,
        "max_batch_size": 32,
        "launch_year": 2020,
        "description": "Datacenter GPU with PCIe form factor",
        "pcie_gen": 4,
        "interconnect_bandwidth_gb_per_sec": 32.0,
        "compute_capability": "8.0",
        "supported_precisions": ["fp32", "tf32", "fp16", "bf16", "int8", "int4"]
    },
    "a100-sxm4-80gb": {
        "name": "NVIDIA A100 SXM4 80GB",
        "family": "Datacenter",
        "gen": "Ampere",
        "vram_gb": 80.0,
        "bandwidth_gb_per_sec": 2039.0,
        "fp32_tflops": 19.5,
        "fp16_tflops": 312.0,
        "bf16_tflops": 312.0,
        "int8_tflops": 624.0,
        "int4_tflops": 1248.0,
        "tdp_watts": 400,
        "tensor_cores": True,
        "max_batch_size": 48,
        "launch_year": 2020,
        "description": "Datacenter GPU with SXM4 form factor and 80GB HBM2e memory",
        "pcie_gen": 4,
        "interconnect_bandwidth_gb_per_sec": 600.0,
        "compute_capability": "8.0",
        "supported_precisions": ["fp32", "tf32", "fp16", "bf16", "int8", "int4"]
    },
    "a10g": {
        "name": "NVIDIA A10G",
        "family": "Datacenter",
        "gen": "Ampere",
        "vram_gb": 24.0,
        "bandwidth_gb_per_sec": 600.0,
        "fp32_tflops": 31.5,
        "fp16_tflops": 125.0,
        "bf16_tflops": 125.0,
        "int8_tflops": 250.0,
        "int4_tflops": 500.0,
        "tdp_watts": 150,
        "tensor_cores": True,
        "max_batch_size": 16,
        "launch_year": 2021,
        "description": "Datacenter GPU for mainstream AI and graphics (Ampere)",
        "pcie_gen": 4,
        "interconnect_bandwidth_gb_per_sec": 32.0,
        "compute_capability": "8.6",
        "supported_precisions": ["fp32", "tf32", "fp16", "bf16", "int8", "int4"]
    },
    
    # Datacenter GPUs - Hopper
    "h100-sxm5-80gb": {
        "name": "NVIDIA H100 SXM5 80GB",
        "family": "Datacenter",
        "gen": "Hopper",
        "vram_gb": 80.0,
        "bandwidth_gb_per_sec": 3350.0,
        "fp32_tflops": 66.9,
        "fp16_tflops": 989.5,
        "bf16_tflops": 989.5,
        "int8_tflops": 1979.0,
        "int4_tflops": 3958.0,
        "tdp_watts": 700,
        "tensor_cores": True,
        "max_batch_size": 64,
        "launch_year": 2022,
        "description": "Flagship datacenter GPU of the Hopper generation with SXM5 form factor",
        "pcie_gen": 5,
        "interconnect_bandwidth_gb_per_sec": 900.0,
        "compute_capability": "9.0",
        "supported_precisions": ["fp32", "tf32", "fp16", "bf16", "fp8", "int8", "int4"]
    },
    "h100-pcie-80gb": {
        "name": "NVIDIA H100 PCIe 80GB",
        "family": "Datacenter",
        "gen": "Hopper",
        "vram_gb": 80.0,
        "bandwidth_gb_per_sec": 2000.0,
        "fp32_tflops": 48.0,
        "fp16_tflops": 660.0,
        "bf16_tflops": 660.0,
        "int8_tflops": 1320.0,
        "int4_tflops": 2640.0,
        "tdp_watts": 350,
        "tensor_cores": True,
        "max_batch_size": 64,
        "launch_year": 2022,
        "description": "PCIe version of the H100 datacenter GPU",
        "pcie_gen": 5,
        "interconnect_bandwidth_gb_per_sec": 64.0,
        "compute_capability": "9.0",
        "supported_precisions": ["fp32", "tf32", "fp16", "bf16", "fp8", "int8", "int4"]
    },
    "h100-pcie-56gb": {
        "name": "NVIDIA H100 PCIe 56GB",
        "family": "Datacenter",
        "gen": "Hopper",
        "vram_gb": 56.0,
        "bandwidth_gb_per_sec": 2000.0,
        "fp32_tflops": 48.0,
        "fp16_tflops": 660.0,
        "bf16_tflops": 660.0,
        "int8_tflops": 1320.0,
        "int4_tflops": 2640.0,
        "tdp_watts": 350,
        "tensor_cores": True,
        "max_batch_size": 48,
        "launch_year": 2022,
        "description": "PCIe version of the H100 datacenter GPU with 56GB memory",
        "pcie_gen": 5,
        "interconnect_bandwidth_gb_per_sec": 64.0,
        "compute_capability": "9.0",
        "supported_precisions": ["fp32", "tf32", "fp16", "bf16", "fp8", "int8", "int4"]
    },
    
    # Newest Datacenter GPUs - Blackwell
    "b100-80gb": {
        "name": "NVIDIA B100 80GB",
        "family": "Datacenter",
        "gen": "Blackwell",
        "vram_gb": 80.0,
        "bandwidth_gb_per_sec": 4500.0,
        "fp32_tflops": 60.0,
        "fp16_tflops": 1000.0,
        "bf16_tflops": 1000.0,
        "int8_tflops": 2000.0,
        "int4_tflops": 4000.0,
        "tdp_watts": 700,
        "tensor_cores": True,
        "max_batch_size": 80,
        "launch_year": 2024,
        "description": "Flagship datacenter GPU of the Blackwell generation",
        "pcie_gen": 5,
        "interconnect_bandwidth_gb_per_sec": 1800.0,
        "compute_capability": "10.0",
        "supported_precisions": ["fp32", "tf32", "fp16", "bf16", "fp8", "int8", "int4"]
    },
    "b200-128gb": {
        "name": "NVIDIA B200 128GB",
        "family": "Datacenter",
        "gen": "Blackwell",
        "vram_gb": 128.0,
        "bandwidth_gb_per_sec": 5000.0,
        "fp32_tflops": 75.0,
        "fp16_tflops": 1200.0,
        "bf16_tflops": 1200.0,
        "int8_tflops": 2400.0,
        "int4_tflops": 4800.0,
        "tdp_watts": 1000,
        "tensor_cores": True,
        "max_batch_size": 96,
        "launch_year": 2024,
        "description": "Top-tier datacenter GPU of the Blackwell generation with expanded memory",
        "pcie_gen": 5,
        "interconnect_bandwidth_gb_per_sec": 1800.0,
        "compute_capability": "10.0",
        "supported_precisions": ["fp32", "tf32", "fp16", "bf16", "fp8", "int8", "int4"]
    },
    
    # Datacenter GPUs - Hopper - High Memory
    "h200-hbm3e-141gb": {
        "name": "NVIDIA H200 HBM3e 141GB",
        "family": "Datacenter",
        "gen": "Hopper",
        "vram_gb": 141.0,
        "bandwidth_gb_per_sec": 4800.0,
        "fp32_tflops": 51.0,
        "fp16_tflops": 989.0,
        "bf16_tflops": 989.0,
        "int8_tflops": 1980.0,
        "int4_tflops": 3960.0,
        "tdp_watts": 700,
        "tensor_cores": True,
        "max_batch_size": 128,
        "launch_year": 2023,
        "description": "Enhanced version of H100 with HBM3e memory for high memory workloads",
        "pcie_gen": 5,
        "interconnect_bandwidth_gb_per_sec": 900.0,
        "compute_capability": "9.0",
        "supported_precisions": ["fp32", "tf32", "fp16", "bf16", "fp8", "int8", "int4"]
    },
    
    # Workstation GPUs
    "rtx-a6000": {
        "name": "NVIDIA RTX A6000",
        "family": "Workstation",
        "gen": "Ampere",
        "vram_gb": 48.0,
        "bandwidth_gb_per_sec": 768.0,
        "fp32_tflops": 38.7,
        "fp16_tflops": 77.4,
        "bf16_tflops": 77.4,
        "int8_tflops": 154.8,
        "int4_tflops": 309.6,
        "tdp_watts": 300,
        "tensor_cores": True,
        "max_batch_size": 32,
        "launch_year": 2020,
        "description": "Professional workstation GPU with high VRAM for creative and scientific workloads",
        "pcie_gen": 4,
        "interconnect_bandwidth_gb_per_sec": 32.0,
        "compute_capability": "8.6",
        "supported_precisions": ["fp32", "fp16", "bf16", "int8", "int4"]
    },
    "rtx-a5000": {
        "name": "NVIDIA RTX A5000",
        "family": "Workstation",
        "gen": "Ampere",
        "vram_gb": 24.0,
        "bandwidth_gb_per_sec": 768.0,
        "fp32_tflops": 27.8,
        "fp16_tflops": 55.6,
        "bf16_tflops": 55.6,
        "int8_tflops": 111.2,
        "int4_tflops": 222.4,
        "tdp_watts": 230,
        "tensor_cores": True,
        "max_batch_size": 16,
        "launch_year": 2021,
        "description": "Professional workstation GPU balanced for creative and AI workloads",
        "pcie_gen": 4,
        "interconnect_bandwidth_gb_per_sec": 32.0,
        "compute_capability": "8.6",
        "supported_precisions": ["fp32", "fp16", "bf16", "int8", "int4"]
    },
    "l40": {
        "name": "NVIDIA L40",
        "family": "Datacenter",
        "gen": "Ada Lovelace",
        "vram_gb": 48.0,
        "bandwidth_gb_per_sec": 864.0,
        "fp32_tflops": 90.5,
        "fp16_tflops": 181.0,
        "bf16_tflops": 181.0,
        "int8_tflops": 362.0,
        "int4_tflops": 724.0,
        "tdp_watts": 300,
        "tensor_cores": True,
        "max_batch_size": 32,
        "launch_year": 2022,
        "description": "Data center GPU optimized for graphics and AI inference",
        "pcie_gen": 4,
        "interconnect_bandwidth_gb_per_sec": 32.0,
        "compute_capability": "8.9",
        "supported_precisions": ["fp32", "fp16", "bf16", "int8", "int4"]
    },
    "l40s": {
        "name": "NVIDIA L40S",
        "family": "Datacenter",
        "gen": "Ada Lovelace",
        "vram_gb": 48.0,
        "bandwidth_gb_per_sec": 864.0,
        "fp32_tflops": 91.6,
        "fp16_tflops": 366.0,
        "bf16_tflops": 366.0,
        "int8_tflops": 733.0,
        "int4_tflops": 733.0,
        "fp8_tflops": 733.0,
        "tdp_watts": 350,
        "tensor_cores": True,
        "max_batch_size": 40,
        "launch_year": 2022,
        "description": "Powerful universal GPU for AI and graphics, Ada Lovelace architecture",
        "pcie_gen": 4,
        "interconnect_bandwidth_gb_per_sec": 64.0,
        "compute_capability": "8.9",
        "supported_precisions": ["fp32", "tf32", "fp16", "bf16", "fp8", "int8", "int4"]
    },
    "b100-sxm6": {
        "name": "NVIDIA B100 SXM6",
        "family": "Datacenter",
        "gen": "Blackwell",
        "vram_gb": 80.0,
        "bandwidth_gb_per_sec": 4500.0,
        "fp32_tflops": 60.0,
        "fp16_tflops": 1000.0,
        "bf16_tflops": 1000.0,
        "int8_tflops": 2000.0,
        "int4_tflops": 4000.0,
        "tdp_watts": 700,
        "tensor_cores": True,
        "max_batch_size": 80,
        "launch_year": 2024,
        "description": "Flagship datacenter GPU of the Blackwell generation with SXM6 form factor",
        "pcie_gen": 5,
        "interconnect_bandwidth_gb_per_sec": 1800.0,
        "compute_capability": "10.0",
        "supported_precisions": ["fp32", "tf32", "fp16", "bf16", "fp8", "int8", "int4"]
    }
}


def get_gpu_config(gpu_name: str) -> Optional[GPUConfig]:
    """
    Get configuration for a specific GPU by name.
    
    Args:
        gpu_name: Name of the GPU to retrieve
        
    Returns:
        GPU configuration dictionary or None if GPU not found
    """
    normalized_name = gpu_name.lower().replace(" ", "-")
    
    # Try direct lookup
    if normalized_name in KNOWN_GPUS:
        return KNOWN_GPUS[normalized_name]
    
    # Try alternative lookups (without hyphens, etc.)
    normalized_alt = normalized_name.replace("-", "")
    for key, gpu in KNOWN_GPUS.items():
        if key.replace("-", "") == normalized_alt:
            return gpu
            
    # Not found
    return None


def get_gpu_families() -> List[str]:
    """
    Get the list of available GPU families.
    
    Returns:
        List of unique GPU families
    """
    return sorted(list(set(gpu["family"] for gpu in KNOWN_GPUS.values())))


def get_gpu_generations() -> List[str]:
    """
    Get the list of available GPU generations.
    
    Returns:
        List of unique GPU generations
    """
    return sorted(list(set(gpu["gen"] for gpu in KNOWN_GPUS.values())))


def get_gpus_by_family(family: str) -> List[GPUConfig]:
    """
    Get all GPUs belonging to a specific family.
    
    Args:
        family: Name of the GPU family
        
    Returns:
        List of GPU configurations in the specified family
    """
    return [gpu for gpu in KNOWN_GPUS.values() if gpu["family"].lower() == family.lower()]


def get_gpus_by_generation(generation: str) -> List[GPUConfig]:
    """
    Get all GPUs belonging to a specific generation.
    
    Args:
        generation: Name of the GPU generation
        
    Returns:
        List of GPU configurations in the specified generation
    """
    return [gpu for gpu in KNOWN_GPUS.values() if gpu["gen"].lower() == generation.lower()]


def get_gpus_by_min_vram(min_vram_gb: float) -> List[GPUConfig]:
    """
    Get all GPUs with at least the specified amount of VRAM.
    
    Args:
        min_vram_gb: Minimum VRAM in gigabytes
        
    Returns:
        List of GPU configurations with sufficient VRAM
    """
    return [gpu for gpu in KNOWN_GPUS.values() if gpu["vram_gb"] >= min_vram_gb]


def get_gpus_supporting_precision(precision: str) -> List[GPUConfig]:
    """
    Get all GPUs that support the specified precision.
    
    Args:
        precision: Precision to check for (e.g., "fp16", "bf16", "fp8")
        
    Returns:
        List of GPU configurations supporting the precision
    """
    return [gpu for gpu in KNOWN_GPUS.values() if precision.lower() in [p.lower() for p in gpu["supported_precisions"]]]


def list_all_gpus() -> List[str]:
    """
    List all available GPU names.
    
    Returns:
        List of GPU names
    """
    return sorted(KNOWN_GPUS.keys())


def get_recommended_gpu_for_model(model_vram_gb: float, min_vram_headroom_gb: float = 2.0) -> List[GPUConfig]:
    """
    Get recommended GPUs for a model with the specified VRAM requirements.
    
    Args:
        model_vram_gb: Model VRAM requirements in gigabytes
        min_vram_headroom_gb: Minimum extra VRAM headroom to recommend
        
    Returns:
        List of GPU configurations that can run the model with sufficient headroom
    """
    required_vram = model_vram_gb + min_vram_headroom_gb
    viable_gpus = get_gpus_by_min_vram(required_vram)
    
    # Sort by efficiency (VRAM per dollar approximation)
    # First datacenter, then workstation, then consumer GPUs
    return sorted(viable_gpus, key=lambda gpu: gpu["vram_gb"]) 