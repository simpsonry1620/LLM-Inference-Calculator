# Advanced Calculator Module

This module provides an advanced, modular calculator for LLM resource estimation.

## Design Philosophy

The Advanced Calculator uses a modular design approach to separate different calculation concerns:

1. **FLOPs Calculations**: Estimating computational requirements for transformer components
2. **VRAM Calculations**: Calculating memory requirements for model weights and runtime
3. **Throughput Calculations**: Predicting tokens per second for inference
4. **Latency Calculations**: Estimating prefill and generation time

This separation offers several benefits:
- Better code organization by functionality
- Easier maintenance and extension 
- Clear separation of concerns
- Independent testing of each component

## Module Structure

- `__init__.py`: Exports the main AdvancedCalculator class
- `main.py`: Main integration class that combines all calculation modules
- `modules/`: Directory containing specialized calculator components
  - `flops.py`: FLOPs calculations for transformer components
  - `vram.py`: Memory requirement estimations
  - `throughput.py`: Throughput predictions
  - `latency.py`: Latency estimations
  - `utils.py`: Shared utilities like validation and history tracking

## Usage Pattern

The main `AdvancedCalculator` class provides a unified interface to all specialized calculators. 
It delegates calculations to the appropriate specialized component internally:

```python
from src.advanced_calculator import AdvancedCalculator

# Create calculator instance
calculator = AdvancedCalculator()

# Calculate FLOPs for attention mechanism (delegates to flops.py)
flops_attention = calculator.calculate_flops_attention(batch_size=32, 
                                                      sequence_length=2048, 
                                                      hidden_dimensions=4096)

# Calculate total VRAM requirements (delegates to vram.py)
total_vram = calculator.calculate_total_vram(batch_size=1,
                                           sequence_length=8192,
                                           hidden_dimensions=8192,
                                           feedforward_dimensions=28672,
                                           num_layers=80,
                                           vocab_size=128000)

# All calculations are logged to history
history = calculator.get_history()
```

## Design Decisions

1. **History Management**: 
   - A shared history system tracks all calculations
   - Each specialized calculator adds entries via callback
   - The main calculator provides history access methods

2. **Validation**:
   - Common validation functions are shared across modules
   - Each function validates its own inputs

3. **Minimal Dependencies**:
   - No external libraries required beyond standard library
   - Focus on clear, understandable formulas over optimization 