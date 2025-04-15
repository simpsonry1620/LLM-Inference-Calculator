"""
Advanced calculator for estimating computational requirements of large language models.

This package provides:
1. FLOPs estimation for transformer operations
2. VRAM requirements for model weights and inference
3. Throughput estimation in tokens per second
4. Latency estimation for prefill and token generation
5. Web interface for easy access to the calculator

Main Classes:
- AdvancedCalculator: Main calculator class that provides all methods
- Web interface via Flask
"""

from .main import AdvancedCalculator
from .web import create_app

__all__ = ['AdvancedCalculator', 'create_app']
