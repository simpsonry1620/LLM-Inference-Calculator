import json
import os
import sys
from typing import List, Dict, Any

# Add the project root to the Python path to allow importing 'src'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.advanced_calculator.main import AdvancedCalculator
except ImportError as e:
    print(f"Error importing AdvancedCalculator: {e}")
    print(
        "Ensure the script is run from the project root or the src directory is in the Python path."
    )
    sys.exit(1)

# --- Benchmark Configuration ---

# From docs/tasks.md table
BENCHMARK_CASES: List[Dict[str, Any]] = [
    # Model, GPU, TTFT Budget, Req/s/GPU (A), GPUs for 10 req/s (B)
    {
        "model": "llama3.1-8b",
        "gpu": "h100-sxm5-80gb",
        "ttft_budget_s": 0.5,
        "bench_req_s_gpu": 4.32,
        "bench_gpus_for_10": 2.32,
    },
    {
        "model": "llama3.1-8b",
        "gpu": "a100-sxm4-80gb",
        "ttft_budget_s": 0.5,
        "bench_req_s_gpu": 0.13,
        "bench_gpus_for_10": 76.8,
    },
    {
        "model": "llama3.1-8b",
        "gpu": "l40s",
        "ttft_budget_s": 0.5,
        "bench_req_s_gpu": 0.077,
        "bench_gpus_for_10": 129.9,
    },
    {
        "model": "llama3.1-8b",
        "gpu": "a10g",
        "ttft_budget_s": 0.5,
        "bench_req_s_gpu": None,
        "bench_gpus_for_10": None,
    },  # N/A
    {
        "model": "llama3.1-8b",
        "gpu": "h100-sxm5-80gb",
        "ttft_budget_s": 3.0,
        "bench_req_s_gpu": 5.41,
        "bench_gpus_for_10": 1.85,
    },
    {
        "model": "llama3.1-8b",
        "gpu": "a100-sxm4-80gb",
        "ttft_budget_s": 3.0,
        "bench_req_s_gpu": 1.33,
        "bench_gpus_for_10": 7.49,
    },
    {
        "model": "llama3.1-8b",
        "gpu": "l40s",
        "ttft_budget_s": 3.0,
        "bench_req_s_gpu": 0.67,
        "bench_gpus_for_10": 15.0,
    },
    {
        "model": "llama3.1-8b",
        "gpu": "a10g",
        "ttft_budget_s": 3.0,
        "bench_req_s_gpu": 0.33,
        "bench_gpus_for_10": 35.45,
    },
    {
        "model": "llama3.1-70b",
        "gpu": "h100-sxm5-80gb",
        "ttft_budget_s": 0.5,
        "bench_req_s_gpu": 0.11,
        "bench_gpus_for_10": 88.4,
    },
    {
        "model": "llama3.1-70b",
        "gpu": "a100-sxm4-80gb",
        "ttft_budget_s": 0.5,
        "bench_req_s_gpu": None,
        "bench_gpus_for_10": None,
    },  # N/A
    {
        "model": "llama3.1-70b",
        "gpu": "l40s",
        "ttft_budget_s": 0.5,
        "bench_req_s_gpu": None,
        "bench_gpus_for_10": None,
    },  # N/A
    {
        "model": "llama3.1-70b",
        "gpu": "a10g",
        "ttft_budget_s": 0.5,
        "bench_req_s_gpu": None,
        "bench_gpus_for_10": None,
    },  # N/A
    {
        "model": "llama3.1-70b",
        "gpu": "h100-sxm5-80gb",
        "ttft_budget_s": 3.0,
        "bench_req_s_gpu": 0.57,
        "bench_gpus_for_10": 17.6,
    },
    {
        "model": "llama3.1-70b",
        "gpu": "a100-sxm4-80gb",
        "ttft_budget_s": 3.0,
        "bench_req_s_gpu": 0.05,
        "bench_gpus_for_10": 197.5,
    },
    {
        "model": "llama3.1-70b",
        "gpu": "l40s",
        "ttft_budget_s": 3.0,
        "bench_req_s_gpu": 0.0057,
        "bench_gpus_for_10": 1768.3,
    },
    {
        "model": "llama3.1-70b",
        "gpu": "a10g",
        "ttft_budget_s": 3.0,
        "bench_req_s_gpu": None,
        "bench_gpus_for_10": None,
    },  # N/A
]

# Common parameters from Task 2
INPUT_SEQUENCE_LENGTH = 5000
OUTPUT_SEQUENCE_LENGTH = 500
BATCH_SIZE = 1
PRECISION = "fp16"
EFFICIENCY_FACTOR = 0.3  # Conservative starting point

# Output file for results
RESULTS_FILE = "logs/benchmark_simulation_results.json"
SKIPPED_FILE = "logs/benchmark_simulation_skipped.json"

# --- Simulation Logic ---


def run_simulations():
    """Runs the benchmark simulations and saves the results."""
    print("Initializing Advanced Calculator...")
    calculator = AdvancedCalculator()
    results = []
    skipped_cases = []

    print(f"Running {len(BENCHMARK_CASES)} benchmark simulations...")

    # Ensure logs directory exists
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    for i, case in enumerate(BENCHMARK_CASES):
        model_name = case["model"]
        gpu_name = case["gpu"]
        print(
            f"  [{i + 1}/{len(BENCHMARK_CASES)}] Simulating: Model={model_name}, GPU={gpu_name}..."
        )

        # Skip cases marked N/A in the benchmark data
        if case["bench_req_s_gpu"] is None:
            print("    Skipping: Benchmark data marked N/A.")
            skipped_info = {"case": case, "reason": "Benchmark data marked N/A"}
            skipped_cases.append(skipped_info)
            continue

        try:
            # Run the analysis
            analysis_result = calculator.analyze_model_on_gpu(
                model_name=model_name,
                gpu_name=gpu_name,
                input_sequence_length=INPUT_SEQUENCE_LENGTH,
                output_sequence_length=OUTPUT_SEQUENCE_LENGTH,
                batch_size=BATCH_SIZE,
                precision=PRECISION,
                efficiency_factor=EFFICIENCY_FACTOR,
            )

            # Combine benchmark case info with calculator results
            combined_result = {
                "benchmark_case": case,
                "calculator_params": {
                    "input_sequence_length": INPUT_SEQUENCE_LENGTH,
                    "output_sequence_length": OUTPUT_SEQUENCE_LENGTH,
                    "batch_size": BATCH_SIZE,
                    "precision": PRECISION,
                    "efficiency_factor": EFFICIENCY_FACTOR,
                },
                "calculator_results": analysis_result,
            }
            results.append(combined_result)
            print(
                f"    Success. TTFT: {analysis_result['performance']['time_to_first_token']:.4f}s"
            )

        except ValueError as e:
            print(f"    Skipping: Encountered ValueError: {e}")
            skipped_info = {"case": case, "reason": f"ValueError: {e}"}
            skipped_cases.append(skipped_info)
        except Exception as e:
            print(f"    Skipping: Encountered unexpected error: {e}")
            skipped_info = {"case": case, "reason": f"Unexpected Error: {e}"}
            skipped_cases.append(skipped_info)

    # Save results
    print(f"Saving results to {RESULTS_FILE}...")
    try:
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=4)
        print("Results saved successfully.")
    except IOError as e:
        print(f"Error saving results: {e}")

    # Save skipped cases
    if skipped_cases:
        print(f"Saving skipped case info to {SKIPPED_FILE}...")
        try:
            with open(SKIPPED_FILE, "w") as f:
                json.dump(skipped_cases, f, indent=4)
            print("Skipped case info saved successfully.")
        except IOError as e:
            print(f"Error saving skipped case info: {e}")


if __name__ == "__main__":
    run_simulations()
