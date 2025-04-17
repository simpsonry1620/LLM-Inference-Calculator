import time
from openai import OpenAI
import sys  # Add sys import
import pathlib  # Add pathlib for path manipulation

# Add project root to the Python path
project_root = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

import tiktoken  # Added for token counting
from src.advanced_calculator.main import AdvancedCalculator  # Added calculator import


# It's good practice to load sensitive data like API keys from environment variables
# Ensure you have NVIDIA_API_KEY set in your environment or a .env file
# If using a different provider, adjust the base_url and key accordingly.

BASE_URL = (
    "http://192.168.50.237:8000/v1"  # Point to API root, not the specific endpoint
)


def benchmark_llm_call(
    model: str = "meta/llama-3.1-8b-instruct",
    prompt: str = "Write a short story about a robot learning to paint.",
    temperature: float = 0.2,
    top_p: float = 0.7,
    max_tokens: int = 1024,
):
    """
    Calls the specified LLM API and benchmarks the time taken for the response stream.

    Args:
        model: The identifier of the model to use.
        prompt: The user prompt to send to the model.
        temperature: Controls randomness in generation. Lower is more deterministic.
        top_p: Nucleus sampling parameter.
        max_tokens: Maximum number of tokens to generate.

    Returns:
        A tuple containing:
        - elapsed_time (float): Time taken in seconds for the API call and stream processing.
        - full_response (str): The complete response string from the LLM.
        - error (str | None): An error message if an exception occurred, otherwise None.
    """
    print(f"--- Benchmarking Model: {model} ---")
    print(f"Prompt: {prompt[:100]}...")  # Print start of prompt for context

    client = OpenAI(
        base_url=BASE_URL,
        api_key="no-key-needed",  # Add dummy key for local/non-auth endpoints
    )

    start_time = time.perf_counter()
    full_response = ""
    error_message = None

    try:
        print("Sending request...")
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=True,
        )

        print("Processing stream...")
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_response += content
                # Optional: print stream progress
                # print(content, end="", flush=True)

        print("Stream finished.")

    except Exception as e:
        print(f"An error occurred: {e}")
        error_message = str(e)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    print(f"Response length: {len(full_response)} characters")
    print("-" * (len(f"--- Benchmarking Model: {model} ---")))  # Match top border

    # Return full results including the model used and prompt for calculator use
    return elapsed_time, full_response, error_message, model, prompt


if __name__ == "__main__":
    # Example usage:
    test_prompt = """\
Explain the concept of Large Language Models in detail.
Cover the following aspects:
1.  What are they and how do they differ from traditional NLP models?
2.  Briefly explain the transformer architecture (self-attention).
3.  What are common training objectives (e.g., masked language modeling, next token prediction)?
4.  Mention some key capabilities (text generation, translation, summarization, question answering).
5.  Discuss some limitations and ethical considerations (bias, hallucinations, computational cost).
Provide the explanation in approximately 150-200 words."""
    time_taken, response, error, model_used, prompt_used = benchmark_llm_call(
        prompt=test_prompt
    )

    if error:
        print(f"Benchmark failed with error: {error}")
    else:
        # --- Add Calculator Comparison ---
        print("\n--- Calculator Comparison ---")
        try:
            # Initialize calculator
            calculator = AdvancedCalculator()
            # Substitute GPU: RTX 6000 Ada Generation (from logs) not found in gpus.py.
            # Using L40S as a high-end Ada Lovelace generation proxy instead.
            # Use the dictionary key ('l40s') for lookup.
            gpu_name = (
                "l40s"  # Was: "NVIDIA L40S" # Was: "NVIDIA RTX 6000 Ada Generation"
            )
            # Adjusting efficiency factor based on previous results
            efficiency_factor = 0.8  # Was: 0.6
            precision = "bf16"  # Updated based on NIM logs (bfloat16) # Was: "fp16" # Common inference precision
            batch_size = 1

            # Estimate token counts using tiktoken
            # Choose an appropriate encoding, e.g., 'cl100k_base' for GPT-4/3.5
            # or find one suitable for Llama 3 if available. Using cl100k as a general default.
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
            except:
                encoding = tiktoken.encoding_for_model("gpt-4")  # Fallback

            input_tokens = len(encoding.encode(prompt_used))
            output_tokens = len(encoding.encode(response))
            print(f"Estimated Input Tokens: {input_tokens}")
            print(f"Estimated Output Tokens: {output_tokens}")

            # Get calculator predictions
            analysis = calculator.analyze_model_on_gpu(
                model_name=model_used,
                gpu_name=gpu_name,
                input_sequence_length=input_tokens,
                output_sequence_length=output_tokens,
                batch_size=batch_size,
                precision=precision,
                efficiency_factor=efficiency_factor,
            )

            predicted_perf = analysis.get("performance", {})
            predicted_time = predicted_perf.get("total_request_time")
            predicted_tps = predicted_perf.get("tokens_per_second")

            # Calculate actual tokens per second (generation phase)
            actual_tps = output_tokens / time_taken if time_taken > 0 else 0

            print("\nBenchmark vs. Calculator:")
            # Check for error key first
            if "error" in predicted_perf:
                print(f"  - Calculator Error: {predicted_perf['error']}")
                print(f"  - Total Time: Actual={time_taken:.4f}s | Predicted=Error")
                print(f"  - Tokens/Sec: Actual={actual_tps:.2f} | Predicted=Error")
            else:
                # Original print logic if no error
                if predicted_time is not None:
                    print(
                        f"  - Total Time: Actual={time_taken:.4f}s | Predicted={predicted_time:.4f}s"
                    )
                else:
                    print(f"  - Total Time: Actual={time_taken:.4f}s | Predicted=N/A")

                if predicted_tps is not None:
                    print(
                        f"  - Tokens/Sec: Actual={actual_tps:.2f} | Predicted={predicted_tps:.2f}"
                    )
                else:
                    print(f"  - Tokens/Sec: Actual={actual_tps:.2f} | Predicted=N/A")

            print(
                "\nNote: Calculator prediction is based on a single GPU ({gpu_name})."
            )
            print("Actual performance is also based on a single GPU setup.")

        except ValueError as calc_error:
            print(f"Calculator analysis failed: {calc_error}")
        except Exception as e:
            print(f"An unexpected error occurred during calculator comparison: {e}")

        # Optionally print the full response
        # print("\nFull Response:")
        # print(response)
        pass  # Response length already printed

    # --- Add more benchmark calls here if needed ---
    # time_taken_2, response_2, error_2, model_used_2, prompt_used_2 = benchmark_llm_call(
    #     model="another/model-id", # Change model if desired
    #     prompt="What is the capital of France?"
    # )
    # Compare for second call if needed...
