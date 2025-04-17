import json
import csv
import os

# Input and Output file paths
RESULTS_JSON_FILE = "logs/benchmark_simulation_results.json"
COMPARISON_CSV_FILE = "logs/comparison_results.csv"

def calculate_metrics(results: dict) -> dict:
    """Calculate Req/s/GPU and GPUs needed for 10 req/s from calculator results."""
    total_request_time = results['calculator_results']['performance']['total_request_time']
    
    # Avoid division by zero if request time is somehow zero or negative
    if total_request_time <= 0:
        calc_req_s_gpu = 0.0
        calc_gpus_for_10 = float('inf')
    else:
        calc_req_s_gpu = 1.0 / total_request_time
        # Avoid division by zero for the next step
        if calc_req_s_gpu == 0:
            calc_gpus_for_10 = float('inf')
        else:
            calc_gpus_for_10 = 10.0 / calc_req_s_gpu
            
    return {
        "calc_req_s_gpu": calc_req_s_gpu,
        "calc_gpus_for_10": calc_gpus_for_10
    }

def convert_results_to_csv():
    """Reads the JSON results and writes a comparison CSV file."""
    print(f"Reading results from {RESULTS_JSON_FILE}...")
    try:
        with open(RESULTS_JSON_FILE, 'r') as f:
            results_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Results file not found at {RESULTS_JSON_FILE}")
        print("Please run the simulation script first (scripts/run_benchmark_simulations.py)")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {RESULTS_JSON_FILE}")
        return
        
    if not results_data:
        print("Result file is empty. No data to convert.")
        return
        
    print(f"Processing {len(results_data)} results...")

    # Define CSV headers
    headers = [
        "Model", "GPU", 
        "Bench_TTFT_Budget_s", "Bench_Req_s_GPU", "Bench_GPUs_for_10",
        "Calc_TTFT_s", "Calc_Total_Time_s", "Calc_Req_s_GPU", "Calc_GPUs_for_10",
        "TTFT_Meets_Budget"
    ]
    
    # Prepare data rows
    rows = []
    for result in results_data:
        bench_case = result["benchmark_case"]
        calc_results = result["calculator_results"]
        calc_perf = calc_results["performance"]
        
        # Perform calculations
        calculated_perf_metrics = calculate_metrics(result)
        
        # Check if TTFT meets budget
        calc_ttft = calc_perf['time_to_first_token']
        bench_budget = bench_case['ttft_budget_s']
        ttft_meets_budget = calc_ttft < bench_budget
        
        row = {
            "Model": bench_case["model"],
            "GPU": bench_case["gpu"],
            "Bench_TTFT_Budget_s": bench_budget,
            "Bench_Req_s_GPU": bench_case["bench_req_s_gpu"],
            "Bench_GPUs_for_10": bench_case["bench_gpus_for_10"],
            "Calc_TTFT_s": calc_ttft,
            "Calc_Total_Time_s": calc_perf['total_request_time'],
            "Calc_Req_s_GPU": calculated_perf_metrics["calc_req_s_gpu"],
            "Calc_GPUs_for_10": calculated_perf_metrics["calc_gpus_for_10"],
            "TTFT_Meets_Budget": ttft_meets_budget
        }
        rows.append(row)

    # Write to CSV
    print(f"Writing comparison data to {COMPARISON_CSV_FILE}...")
    try:
        # Ensure logs directory exists
        os.makedirs(os.path.dirname(COMPARISON_CSV_FILE), exist_ok=True)
        
        with open(COMPARISON_CSV_FILE, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        print("CSV file created successfully.")
    except IOError as e:
        print(f"Error writing CSV file: {e}")

if __name__ == "__main__":
    convert_results_to_csv() 