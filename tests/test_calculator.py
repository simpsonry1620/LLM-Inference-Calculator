"""
Tests for the LLM infrastructure scaling calculator.
"""
import pytest
import sys
import os

# Add parent directory to path to import the calculator module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.calculator import LLMScalingCalculator
from advanced_calculator import AdvancedCalculator


def test_calculator_initialization():
    """Test calculator initialization with default values."""
    calculator = LLMScalingCalculator()
    assert calculator.gpu_memory_gb == 80
    assert calculator.gpu_flops == 312e12
    assert calculator.gpu_cost_per_hour == 1.5


def test_memory_requirements():
    """Test memory requirements calculation."""
    calculator = LLMScalingCalculator()
    
    # Test with 1B parameter model
    memory_reqs = calculator.estimate_memory_requirements(
        model_size_params=1e9,
        batch_size=32,
        sequence_length=2048
    )
    
    # Model size in GB (1B * 2 bytes / GB conversion)
    expected_model_size_gb = 1e9 * 2 / (1024**3)
    assert memory_reqs["model_size_gb"] == pytest.approx(expected_model_size_gb)
    
    # Optimizer should be 12x model size
    assert memory_reqs["optimizer_size_gb"] == pytest.approx(expected_model_size_gb * 12)
    
    # Total memory should be sum of components
    total_memory = (
        memory_reqs["model_size_gb"] + 
        memory_reqs["optimizer_size_gb"] + 
        memory_reqs["activation_size_gb"]
    )
    assert memory_reqs["total_memory_gb"] == pytest.approx(total_memory)
    
    # Check GPUs needed calculation
    expected_gpus = max(1, int(memory_reqs["total_memory_gb"] / calculator.gpu_memory_gb))
    assert memory_reqs["gpus_needed"] == expected_gpus


def test_compute_requirements():
    """Test compute requirements calculation."""
    calculator = LLMScalingCalculator()
    
    # Test with 1B parameter model, simple values for easy verification
    compute_reqs = calculator.estimate_compute_requirements(
        model_size_params=1e9,
        batch_size=32,
        sequence_length=1024,
        tokens_to_train=1e9  # 1B tokens for easy math
    )
    
    # FLOPs per token should be 6 * params * seq_len
    expected_flops_per_token = 6 * 1e9 * 1024
    assert compute_reqs["flops_per_token"] == pytest.approx(expected_flops_per_token)
    
    # Total FLOPs should be flops_per_token * tokens
    expected_total_flops = expected_flops_per_token * 1e9
    assert compute_reqs["total_flops"] == pytest.approx(expected_total_flops)


def test_end_to_end_estimation():
    """Test end-to-end resource estimation."""
    calculator = LLMScalingCalculator()
    
    # Test with 7B parameter model (typical small LLM)
    results = calculator.estimate_resources(
        model_size_params=7e9,
        batch_size=32,
        sequence_length=2048,
        tokens_to_train=300e9
    )
    
    # Check that all sections are present
    assert "memory_requirements" in results
    assert "compute_requirements" in results
    assert "summary" in results
    
    # Check that summary values are consistent with detailed results
    summary = results["summary"]
    memory = results["memory_requirements"]
    compute = results["compute_requirements"]
    
    assert summary["model_size_billions"] == 7
    assert summary["gpus_needed"] == memory["gpus_needed"]
    assert summary["training_days"] == compute["training_days"]
    assert summary["training_cost_usd"] == compute["training_cost_usd"]


def test_advanced_calculator():
    """Test the Advanced Calculator functionality."""
    # Llama 3.1 70B parameters
    batch = 1  # Typical inference batch size
    seq_len = 8192  # Context length capability
    hidden_dim = 8192  # Estimated hidden dimension
    ff_dim = 28672  # Estimated feedforward dimension (3.5x hidden_dim)
    num_layers = 80  # Estimated number of layers
    vocab_size = 128000  # Estimated vocabulary size

    calc = AdvancedCalculator()
    
    # Test attention FLOPs
    flops_attention = calc.calculate_flops_attention(batch, seq_len, hidden_dim)
    expected_attention_flops = batch * seq_len * (hidden_dim**2 + seq_len * hidden_dim)
    assert flops_attention == pytest.approx(expected_attention_flops)
    assert flops_attention > 0
    print(f"Llama 3.1 70B Attention FLOPs: {flops_attention:,}")
    
    # Test feedforward FLOPs
    flops_feedforward = calc.calculate_flops_feedforward(batch, seq_len, hidden_dim, ff_dim)
    expected_feedforward_flops = 2 * batch * seq_len * hidden_dim * ff_dim
    assert flops_feedforward == pytest.approx(expected_feedforward_flops)
    assert flops_feedforward > 0
    print(f"Llama 3.1 70B Feedforward FLOPs: {flops_feedforward:,}")
    
    # Test prefill FLOPs
    flops_prefill = calc.calculate_flops_prefill(batch, seq_len, hidden_dim, ff_dim, num_layers)
    expected_prefill_flops = num_layers * (flops_attention + flops_feedforward)
    assert flops_prefill == pytest.approx(expected_prefill_flops)
    assert flops_prefill > 0
    print(f"Llama 3.1 70B Prefill FLOPs: {flops_prefill:,}")
    
    # Test VRAM calculation
    model_vram_fp16 = calc.calculate_model_vram(hidden_dim, ff_dim, num_layers, vocab_size, precision="fp16")
    
    # Calculate expected VRAM manually using the same formula as in the calculator
    attention_params = 4 * hidden_dim * hidden_dim
    ff_params = 2 * hidden_dim * ff_dim
    layer_norm_params = 4 * hidden_dim
    params_per_layer = attention_params + ff_params + layer_norm_params
    total_layer_params = num_layers * params_per_layer
    embedding_params = vocab_size * hidden_dim
    final_layer_norm = 2 * hidden_dim
    total_params = total_layer_params + embedding_params + final_layer_norm
    bytes_per_param = 2  # fp16
    expected_vram_fp16 = (total_params * bytes_per_param) / (1024**3)
    
    assert model_vram_fp16 == pytest.approx(expected_vram_fp16, rel=1e-5)
    assert model_vram_fp16 > 0
    print(f"Llama 3.1 70B VRAM (FP16): {model_vram_fp16:.2f} GB")
    
    # Test FP32 precision
    model_vram_fp32 = calc.calculate_model_vram(hidden_dim, ff_dim, num_layers, vocab_size, precision="fp32")
    assert model_vram_fp32 == pytest.approx(model_vram_fp16 * 2, rel=1e-5)
    print(f"Llama 3.1 70B VRAM (FP32): {model_vram_fp32:.2f} GB")
    
    # Test A100 throughput calculation
    a100_tflops = 312  # A100 theoretical peak performance in TFLOPS (FP16)
    theoretical_tokens_per_second = (a100_tflops * 10**12) / (flops_prefill / seq_len)
    print(f"Theoretical tokens/second on A100 (at 100% efficiency): {theoretical_tokens_per_second:.2f}")
    print(f"Realistic tokens/second (at 30% efficiency): {theoretical_tokens_per_second * 0.3:.2f}")


def test_validation():
    """Test that input validation works correctly."""
    calc = AdvancedCalculator()
    
    # Test with invalid inputs
    with pytest.raises(TypeError):
        calc.calculate_flops_attention("not_an_int", 512, 768)
        
    with pytest.raises(ValueError):
        calc.calculate_flops_attention(-1, 512, 768)
        
    with pytest.raises(TypeError):
        calc.calculate_model_vram(768, 3072, 12, "not_an_int")


def test_history():
    """Test that the history is correctly maintained."""
    calc = AdvancedCalculator()
    
    # Clear history
    calc.clear_history()
    assert len(calc.get_history()) == 0
    
    # Perform a calculation
    calc.calculate_model_vram(8192, 28672, 80, 128000)
    
    # Check that history is updated
    history = calc.get_history()
    assert len(history) == 1
    assert "Model_VRAM" in history[0]
    assert "8192" in history[0]


if __name__ == "__main__":
    pytest.main() 