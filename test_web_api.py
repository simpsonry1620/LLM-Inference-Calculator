import requests
import json
import time

def test_web_api():
    # Give the server time to start
    time.sleep(2)
    
    # API endpoint
    url = "http://localhost:5000/api/calculate"
    
    # Test data
    data = {
        "model_name": "llama3-8b",
        "hidden_dim": 4096,
        "ff_dim": 14336,
        "num_layers": 32,
        "vocab_size": 128000,
        "seq_len": 1000,
        "batch_size": 1,
        "gpu_tflops": 989,
        "efficiency": 0.3,
        "precision": "fp16",
        "parallelism_strategy": "none",
        "tp_size": 1,
        "pp_size": 1,
        "num_gpus": 1,
        "gpu_id": "h200-hbm3e-141gb"
    }
    
    # Call the API
    try:
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            result = response.json()
            
            # Check if parallelism field is present
            if "parallelism" in result:
                print("Success: Parallelism field is present")
                print(f"Parallelism strategy: {result['parallelism']['strategy']}")
                print(f"TP size: {result['parallelism']['tp_size']}")
                print(f"PP size: {result['parallelism']['pp_size']}")
                print(f"Number of GPUs: {result['parallelism']['num_gpus']}")
            else:
                print("Error: Parallelism field is missing!")
                
            # Check if overheads_used field is present
            if "overheads_used" in result:
                print("\nSuccess: Overheads_used field is present")
                print(f"Weights overhead: {result['overheads_used']['weights']}")
                print(f"KV cache overhead: {result['overheads_used']['kv_cache']}")
                print(f"Activations overhead: {result['overheads_used']['activations']}")
                print(f"System overhead: {result['overheads_used']['system']}")
            else:
                print("Error: Overheads_used field is missing!")
                
        else:
            print(f"Error: HTTP {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Exception: {str(e)}")

if __name__ == "__main__":
    test_web_api() 