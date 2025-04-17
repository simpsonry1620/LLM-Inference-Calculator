import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext # Add scrolledtext
import sys
import os
import traceback
import json # Add json for formatting the dict

# Adjust path to import from src
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, project_root)

from src.advanced_calculator.main import AdvancedCalculator

# --- Global Variables ---
calculator = AdvancedCalculator()

# --- Helper Functions ---
def get_int_from_entry(entry_widget, default_value=0):
    """Safely get an integer from an entry widget."""
    try:
        value = entry_widget.get()
        # Allow empty string to mean default value
        return int(value) if value else default_value
    except ValueError:
        messagebox.showerror("Input Error", f"Invalid integer input: '{value}'")
        return None # Indicate error

def get_float_from_entry(entry_widget, default_value=0.0):
    """Safely get a float from an entry widget."""
    try:
        value = entry_widget.get()
        # Allow empty string to mean default value
        return float(value) if value else default_value
    except ValueError:
        messagebox.showerror("Input Error", f"Invalid float input: '{value}'")
        return None # Indicate error

# --- Event Handlers ---
def update_tflops_entry(event, widgets):
    """Auto-fill TFLOPS entry based on selected GPU and Precision."""
    gpu_name = widgets['gpu_var'].get()
    precision = widgets['precision_var'].get().lower()
    tflops_entry = widgets['gpu_tflops_entry']

    if gpu_name == "Custom" or not gpu_name:
        tflops_entry.config(state=tk.NORMAL)
        # Don't clear if custom, user might be typing
        return

    gpu_config = calculator.get_gpu_config(gpu_name)
    if not gpu_config:
        tflops_entry.config(state=tk.NORMAL)
        tflops_entry.delete(0, tk.END)
        tflops_entry.insert(0, "N/A")
        return

    # Determine the correct TFLOPS key
    tflops_key = None
    if precision == "fp16":
        tflops_key = "fp16_tflops"
    elif precision == "bf16":
        tflops_key = "bf16_tflops" if "bf16_tflops" in gpu_config else "fp16_tflops"
    elif precision == "fp32":
        tflops_key = "fp32_tflops"
    elif precision == "int8":
        tflops_key = "int8_tflops"
    elif precision == "int4":
         tflops_key = "int4_tflops"

    tflops_value = None
    if tflops_key and tflops_key in gpu_config:
        tflops_value = gpu_config[tflops_key]

    tflops_entry.config(state=tk.NORMAL) # Ensure entry is editable before modification
    tflops_entry.delete(0, tk.END)
    if tflops_value is not None:
        # Correctly insert the found value
        tflops_entry.insert(0, str(tflops_value))
    else:
        tflops_entry.insert(0, "N/A") # Indicate TFLOPS not available for this precision

def calculate_requirements(widgets):
    """Reads inputs, performs calculations, and updates result labels."""
    status_label = widgets['status_label']
    status_label.config(text="Status: Calculating...")
    root = status_label.winfo_toplevel()
    root.update_idletasks()

    # Clear previous results
    for key in ['total_vram_value', 'weights_vram_value', 'kv_cache_vram_value',
                'activations_vram_value', 'system_overhead_value', 'flops_per_token_value',
                'prefill_latency_value', 'token_latency_value', 'ttft_value',
                'trt_value', 'tokens_per_sec_value',
                'parallelism_strategy_value', 'parallelism_num_gpus_value',
                'parallelism_tp_degree_value', 'parallelism_pp_degree_value',
                'parallelism_vram_per_gpu_value']:
        if key in widgets:
             widgets[key].config(text="-")

    # Clear parallelism text box
    if 'parallelism_text' in widgets:
        widgets['parallelism_text'].config(state=tk.NORMAL)
        widgets['parallelism_text'].delete('1.0', tk.END)
        widgets['parallelism_text'].config(state=tk.DISABLED)

    selected_model = widgets['model_var'].get()
    use_predefined_model = selected_model != "Custom"

    model_params = {}

    try:
        if use_predefined_model:
            config = calculator.get_model_config(selected_model)
            if not config:
                raise ValueError(f"Could not find config for model {selected_model}")
            model_params = {
                'hidden_dimensions': config.get('hidden_dimensions'),
                'feedforward_dimensions': config.get('feedforward_dimensions'),
                'num_layers': config.get('num_layers'),
                'vocab_size': config.get('vocab_size')
            }
            # Update entry fields (ensure they are enabled first)
            for key in ['hidden_dim_entry', 'ff_dim_entry', 'num_layers_entry', 'vocab_size_entry']:
                widgets[key].config(state=tk.NORMAL)

            widgets['hidden_dim_entry'].delete(0, tk.END)
            widgets['hidden_dim_entry'].insert(0, str(model_params.get('hidden_dimensions', '')))
            widgets['ff_dim_entry'].delete(0, tk.END)
            widgets['ff_dim_entry'].insert(0, str(model_params.get('feedforward_dimensions', '')))
            widgets['num_layers_entry'].delete(0, tk.END)
            widgets['num_layers_entry'].insert(0, str(model_params.get('num_layers', '')))
            widgets['vocab_size_entry'].delete(0, tk.END)
            widgets['vocab_size_entry'].insert(0, str(model_params.get('vocab_size', '')))

            for key in ['hidden_dim_entry', 'ff_dim_entry', 'num_layers_entry', 'vocab_size_entry']:
                 widgets[key].config(state=tk.DISABLED)

        else: # Custom model parameters
            model_params['hidden_dimensions'] = get_int_from_entry(widgets['hidden_dim_entry'])
            model_params['feedforward_dimensions'] = get_int_from_entry(widgets['ff_dim_entry'])
            model_params['num_layers'] = get_int_from_entry(widgets['num_layers_entry'])
            model_params['vocab_size'] = get_int_from_entry(widgets['vocab_size_entry'])

        # Get common parameters
        batch_size = get_int_from_entry(widgets['batch_size_entry'], default_value=1)
        input_sequence_length = get_int_from_entry(widgets['input_seq_len_entry'], default_value=2048)
        output_sequence_length = get_int_from_entry(widgets['output_seq_len_entry'], default_value=128)
        precision = widgets['precision_var'].get()
        gpu_tflops_str = widgets['gpu_tflops_entry'].get()
        # Handle potential "N/A" from auto-fill
        gpu_tflops = 0.0
        if gpu_tflops_str.lower() != 'n/a':
             gpu_tflops = get_float_from_entry(widgets['gpu_tflops_entry'], default_value=0.0)
        else:
            # Try to get TFLOPS from selected GPU if not custom
            selected_gpu = widgets['gpu_var'].get()
            if selected_gpu != "Custom":
                gpu_config = calculator.get_gpu_config(selected_gpu)
                if gpu_config:
                    tflops_key = None
                    prec_lower = precision.lower()
                    if prec_lower == "fp16": tflops_key = "fp16_tflops"
                    elif prec_lower == "bf16": tflops_key = "bf16_tflops" if "bf16_tflops" in gpu_config else "fp16_tflops"
                    elif prec_lower == "fp32": tflops_key = "fp32_tflops"
                    elif prec_lower == "int8": tflops_key = "int8_tflops"
                    elif prec_lower == "int4": tflops_key = "int4_tflops"

                    if tflops_key and tflops_key in gpu_config:
                         gpu_tflops = gpu_config[tflops_key]
                         widgets['gpu_tflops_entry'].config(state=tk.NORMAL)
                         widgets['gpu_tflops_entry'].delete(0, tk.END)
                         widgets['gpu_tflops_entry'].insert(0, str(gpu_tflops))
                         widgets['gpu_tflops_entry'].config(state=tk.DISABLED if selected_gpu != "Custom" else tk.NORMAL) # Re-disable if needed
                         status_label.config(text="Status: Found TFLOPS, Calculating...")
                         root.update_idletasks()
                    else:
                         raise ValueError(f"TFLOPS for {selected_gpu} @ {precision} not found in config.")
                else:
                     raise ValueError(f"GPU config for {selected_gpu} not found.")
            else: # Custom GPU and TFLOPS is N/A
                raise ValueError("GPU TFLOPS not available or specified for Custom GPU.")

        efficiency_pct = get_float_from_entry(widgets['efficiency_entry'], default_value=30.0)
        efficiency_factor = efficiency_pct / 100.0 if efficiency_pct is not None else None

        # Check if any input conversion failed
        input_values = [
            model_params.get('hidden_dimensions'), model_params.get('feedforward_dimensions'),
            model_params.get('num_layers'), model_params.get('vocab_size'),
            batch_size, input_sequence_length, output_sequence_length,
            gpu_tflops, efficiency_factor # Add new performance inputs
        ]
        if None in input_values:
             raise ValueError("Invalid input detected.")
        
        if gpu_tflops <= 0:
            raise ValueError("GPU TFLOPS must be positive and available.")
        if not (0 < efficiency_factor <= 1.0):
            raise ValueError("Efficiency factor must be between 0 and 100%.")

        # Ensure FF dim is present (Estimate if missing)
        if not model_params.get('feedforward_dimensions') and model_params.get('hidden_dimensions'):
             model_params['feedforward_dimensions'] = model_params['hidden_dimensions'] * 4
             status_label.config(text="Status: Estimated FF Dim, Calculating...")
             root.update_idletasks()

        # --- Perform Calculations ---
        # 1. VRAM
        vram_results = calculator.calculate_total_vram(
            batch_size=batch_size,
            # Pass input and output lengths separately as expected by AdvancedCalculator
            input_sequence_length=input_sequence_length,
            output_sequence_length=output_sequence_length,
            hidden_dimensions=model_params['hidden_dimensions'],
            feedforward_dimensions=model_params['feedforward_dimensions'],
            num_layers=model_params['num_layers'],
            vocab_size=model_params['vocab_size'],
            precision=precision
        )

        # 2. FLOPs
        flops_per_token = calculator.calculate_flops_per_token(
            batch_size=batch_size,
            hidden_dimensions=model_params['hidden_dimensions'],
            feedforward_dimensions=model_params['feedforward_dimensions'],
            num_layers=model_params['num_layers']
        )
        flops_prefill = calculator.calculate_flops_prefill(
            batch_size=batch_size,
            sequence_length=input_sequence_length, # Prefill uses input length
            hidden_dimensions=model_params['hidden_dimensions'],
            feedforward_dimensions=model_params['feedforward_dimensions'],
            num_layers=model_params['num_layers']
        )

        # 3. Latency & Throughput
        prefill_latency_s = calculator.calculate_prefill_latency(flops_prefill, gpu_tflops, efficiency_factor)
        token_latency_s = calculator.calculate_token_latency(flops_per_token, gpu_tflops, efficiency_factor)
        tokens_per_sec = calculator.estimate_inference_throughput(flops_per_token, gpu_tflops, efficiency_factor)

        # Calculate derived metrics
        ttft_s = prefill_latency_s
        trt_s = prefill_latency_s + (output_sequence_length * token_latency_s)

        # --- Update Result Labels ---
        # VRAM Results
        total_vram_required_gb = vram_results.get('total', 0.0)
        widgets['total_vram_value'].config(text=f"{total_vram_required_gb:.2f} GB")
        widgets['weights_vram_value'].config(text=f"{vram_results.get('weights_with_overhead', 0.0):.2f} GB")
        widgets['kv_cache_vram_value'].config(text=f"{vram_results.get('kv_cache_with_overhead', 0.0):.2f} GB")
        widgets['activations_vram_value'].config(text=f"{vram_results.get('activations_with_overhead', 0.0):.2f} GB")
        widgets['system_overhead_value'].config(text=f"{vram_results.get('system_overhead_applied', 0.0):.2f} GB")
        
        # FLOPs Results
        widgets['flops_per_token_value'].config(text=f"{flops_per_token / 1e9:.2f} GFLOPs")

        # Latency/Throughput Results (Convert s to ms where appropriate)
        widgets['prefill_latency_value'].config(text=f"{prefill_latency_s * 1000:.2f} ms")
        widgets['token_latency_value'].config(text=f"{token_latency_s * 1000:.3f} ms")
        widgets['ttft_value'].config(text=f"{ttft_s * 1000:.2f} ms")
        widgets['trt_value'].config(text=f"{trt_s * 1000:.2f} ms")
        widgets['tokens_per_sec_value'].config(text=f"{tokens_per_sec:.2f}")

        # --- Parallelism Check ---
        scaling_info = None
        selected_gpu_name = widgets['gpu_var'].get()
        gpu_vram_gb = 0.0

        print(f"[DEBUG] Selected GPU: {selected_gpu_name}") # DEBUG

        if selected_gpu_name != "Custom":
            gpu_config = calculator.get_gpu_config(selected_gpu_name)
            if gpu_config and 'vram_gb' in gpu_config:
                gpu_vram_gb = float(gpu_config['vram_gb'])
                print(f"[DEBUG] GPU VRAM: {gpu_vram_gb} GB") # DEBUG
                print(f"[DEBUG] Required VRAM: {total_vram_required_gb:.2f} GB") # DEBUG

                if total_vram_required_gb > gpu_vram_gb:
                    print("[DEBUG] Required VRAM > GPU VRAM. Attempting scaling calculation...") # DEBUG
                    status_label.config(text="Status: Determining scaling...")
                    root.update_idletasks()
                    try:
                        # Use the sequence length for the scaling calculation
                        # The method calculates total VRAM internally
                        # Pass combined length for peak VRAM scenario
                        scaling_sequence_length = input_sequence_length + output_sequence_length
                        scaling_info = calculator.determine_model_scaling(
                            gpu_vram_gb=gpu_vram_gb,
                            batch_size=batch_size,
                            sequence_length=scaling_sequence_length,
                            hidden_dimensions=model_params['hidden_dimensions'],
                            feedforward_dimensions=model_params['feedforward_dimensions'],
                            num_layers=model_params['num_layers'],
                            vocab_size=model_params['vocab_size'],
                            precision=precision
                        )
                        print(f"[DEBUG] Scaling Info Result: {scaling_info}") # DEBUG
                    except Exception as scale_e:
                         messagebox.showerror("Scaling Error", f"Could not determine scaling: {scale_e}")
                         status_label.config(text=f"Status: Scaling Error - {scale_e}")
                         # Don't stop, just show VRAM results
                         scaling_info = None # Ensure it's None
            else:
                 messagebox.showwarning("GPU Info Missing", f"VRAM info for selected GPU '{selected_gpu_name}' not found. Cannot determine scaling.")
        else:
            # If Custom GPU is selected, we can't automatically determine scaling
            # These individual labels don't exist; info goes into the text box below.
            # widgets['parallelism_strategy_value'].config(text="N/A (Custom GPU)")
            # widgets['parallelism_num_gpus_value'].config(text="-")
            # widgets['parallelism_tp_degree_value'].config(text="-")
            # widgets['parallelism_pp_degree_value'].config(text="-")
            # widgets['parallelism_vram_per_gpu_value'].config(text="-")
            pass # No individual labels to update here

        # Update Parallelism Labels / Text Box
        widgets['parallelism_text'].config(state=tk.NORMAL)
        widgets['parallelism_text'].delete('1.0', tk.END) # Clear previous text
        if scaling_info:
             # Format the dictionary nicely
             scaling_str = json.dumps(scaling_info, indent=2)
             widgets['parallelism_text'].insert(tk.END, scaling_str)
        elif selected_gpu_name != "Custom" and gpu_vram_gb > 0:
            # Fits on single GPU
             widgets['parallelism_text'].insert(tk.END, "Model fits on single selected GPU.")
        elif selected_gpu_name == "Custom":
             widgets['parallelism_text'].insert(tk.END, "N/A (Select a specific GPU to determine scaling)")
        else: # Predefined GPU but VRAM info missing
            widgets['parallelism_text'].insert(tk.END, "Could not determine scaling (Missing VRAM info for selected GPU).")
        widgets['parallelism_text'].config(state=tk.DISABLED)

        status_label.config(text="Status: Done")

    except Exception as e:
        # Log traceback to console for debugging, show simple error in GUI
        print("--- ERROR ---")
        traceback.print_exc()
        print("-------------")
        messagebox.showerror("Calculation Error", f"An error occurred: {e}")
        status_label.config(text=f"Status: Error - {e}")

    widgets['status_label'].config(text="Status: Ready")

def on_model_select(event, widgets):
    print("[DEBUG] on_model_select START") # DEBUG
    """Handles model selection changes."""
    selected_model = widgets['model_var'].get()
    is_custom = selected_model == "Custom"

    # Define which fields are part of the model config
    model_config_fields = ['hidden_dim_entry', 'ff_dim_entry', 'num_layers_entry', 'vocab_size_entry']

    for field_key in model_config_fields:
        widgets[field_key].config(state=(tk.NORMAL if is_custom else tk.DISABLED))

    if not is_custom:
        config = calculator.get_model_config(selected_model)
        if config:
            # Ensure fields are normal before clearing/inserting
            for field_key in model_config_fields:
                 widgets[field_key].config(state=tk.NORMAL)
            # Clear and insert into enabled fields
            widgets['hidden_dim_entry'].delete(0, tk.END)
            widgets['hidden_dim_entry'].insert(0, str(config.get('hidden_dimensions', '')))
            widgets['ff_dim_entry'].delete(0, tk.END)
            widgets['ff_dim_entry'].insert(0, str(config.get('feedforward_dimensions', '')))
            widgets['num_layers_entry'].delete(0, tk.END)
            widgets['num_layers_entry'].insert(0, str(config.get('num_layers', '')))
            widgets['vocab_size_entry'].delete(0, tk.END)
            widgets['vocab_size_entry'].insert(0, str(config.get('vocab_size', '')))
            # Disable fields after filling
            for field_key in model_config_fields:
                widgets[field_key].config(state=tk.DISABLED)
    else:
         # Clear fields when switching to custom
        for field_key in model_config_fields:
             widgets[field_key].config(state=tk.NORMAL) # Ensure enabled before clearing
             widgets[field_key].delete(0, tk.END)

    # Clear results when model changes
    for key in ['total_vram_value', 'weights_vram_value', 'kv_cache_vram_value',
                'activations_vram_value', 'system_overhead_value', 'flops_per_token_value',
                'prefill_latency_value', 'token_latency_value', 'ttft_value',
                'trt_value', 'tokens_per_sec_value',
                 # Parallelism handled by text box, these keys are not used for individual labels
                # 'parallelism_strategy_value', 'parallelism_num_gpus_value',
                # 'parallelism_tp_degree_value', 'parallelism_pp_degree_value',
                # 'parallelism_vram_per_gpu_value'
                ]:
         if key in widgets:
            widgets[key].config(text="-")
    widgets['status_label'].config(text="Status: Ready")

    # Reset GPU selection and TFLOPS when model changes
    # if 'gpu_var' in widgets:
    #     widgets['gpu_var'].set("Custom")
    #     widgets['gpu_tflops_entry'].config(state=tk.NORMAL)
    #     widgets['gpu_tflops_entry'].delete(0, tk.END)
    #     # Optionally set a default TFLOPS like A100 FP16?
    #     widgets['gpu_tflops_entry'].insert(0, "312")

    print("[DEBUG] on_model_select END") # DEBUG

# --- GUI Setup ---
def main():
    root = tk.Tk()
    root.title("LLM Inference Calculator GUI")

    main_frame = ttk.Frame(root, padding="10 10 10 10")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    widgets = {}

    # --- Input Fields Frame ---
    input_frame = ttk.LabelFrame(main_frame, text="Parameters", padding="10")
    input_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
    input_frame.columnconfigure(1, weight=1)

    row_index = 0
    # Model Selection
    ttk.Label(input_frame, text="Model:").grid(column=0, row=row_index, sticky=tk.W, padx=5, pady=5)
    model_var = tk.StringVar()
    widgets['model_var'] = model_var
    model_names = ["Custom"] + calculator.get_available_models()
    model_combobox = ttk.Combobox(input_frame, textvariable=model_var, values=model_names, width=25, state='readonly') # Wider combobox
    model_combobox.grid(column=1, row=row_index, sticky=(tk.W, tk.E), padx=5, pady=5)
    model_combobox.set("Custom")
    model_combobox.bind('<<ComboboxSelected>>', lambda event: on_model_select(event, widgets))
    widgets['model_combobox'] = model_combobox
    row_index += 1

    # Hidden Dimensions
    ttk.Label(input_frame, text="Hidden Dimensions:").grid(column=0, row=row_index, sticky=tk.W, padx=5, pady=5)
    hidden_dim_entry = ttk.Entry(input_frame, width=20)
    hidden_dim_entry.grid(column=1, row=row_index, sticky=(tk.W, tk.E), padx=5, pady=5)
    widgets['hidden_dim_entry'] = hidden_dim_entry
    row_index += 1

    # Feedforward Dimensions
    ttk.Label(input_frame, text="Feedforward Dim:").grid(column=0, row=row_index, sticky=tk.W, padx=5, pady=5)
    ff_dim_entry = ttk.Entry(input_frame, width=20)
    ff_dim_entry.grid(column=1, row=row_index, sticky=(tk.W, tk.E), padx=5, pady=5)
    widgets['ff_dim_entry'] = ff_dim_entry
    row_index += 1

    # Number of Layers
    ttk.Label(input_frame, text="Number of Layers:").grid(column=0, row=row_index, sticky=tk.W, padx=5, pady=5)
    num_layers_entry = ttk.Entry(input_frame, width=20)
    num_layers_entry.grid(column=1, row=row_index, sticky=(tk.W, tk.E), padx=5, pady=5)
    widgets['num_layers_entry'] = num_layers_entry
    row_index += 1

    # Vocabulary Size
    ttk.Label(input_frame, text="Vocabulary Size:").grid(column=0, row=row_index, sticky=tk.W, padx=5, pady=5)
    vocab_size_entry = ttk.Entry(input_frame, width=20)
    vocab_size_entry.grid(column=1, row=row_index, sticky=(tk.W, tk.E), padx=5, pady=5)
    widgets['vocab_size_entry'] = vocab_size_entry
    row_index += 1

    # Input Sequence Length
    ttk.Label(input_frame, text="Input Sequence Len:").grid(column=0, row=row_index, sticky=tk.W, padx=5, pady=5)
    input_seq_len_entry = ttk.Entry(input_frame, width=20)
    input_seq_len_entry.insert(0, "2048")
    input_seq_len_entry.grid(column=1, row=row_index, sticky=(tk.W, tk.E), padx=5, pady=5)
    widgets['input_seq_len_entry'] = input_seq_len_entry
    row_index += 1

    # Output Sequence Length
    ttk.Label(input_frame, text="Output Sequence Len:").grid(column=0, row=row_index, sticky=tk.W, padx=5, pady=5)
    output_seq_len_entry = ttk.Entry(input_frame, width=20)
    output_seq_len_entry.insert(0, "128")
    output_seq_len_entry.grid(column=1, row=row_index, sticky=(tk.W, tk.E), padx=5, pady=5)
    widgets['output_seq_len_entry'] = output_seq_len_entry
    row_index += 1

    # Batch Size
    ttk.Label(input_frame, text="Batch Size:").grid(column=0, row=row_index, sticky=tk.W, padx=5, pady=5)
    batch_size_entry = ttk.Entry(input_frame, width=20)
    batch_size_entry.insert(0, "1")
    batch_size_entry.grid(column=1, row=row_index, sticky=(tk.W, tk.E), padx=5, pady=5)
    widgets['batch_size_entry'] = batch_size_entry
    row_index += 1

    # Precision
    ttk.Label(input_frame, text="Precision:").grid(column=0, row=row_index, sticky=tk.W, padx=5, pady=5)
    precision_var = tk.StringVar()
    widgets['precision_var'] = precision_var
    precision_values = ["fp16", "bf16", "fp32", "int8", "int4"] # Added int4
    precision_combobox = ttk.Combobox(input_frame, textvariable=precision_var, values=precision_values, width=18, state='readonly')
    precision_combobox.grid(column=1, row=row_index, sticky=(tk.W, tk.E), padx=5, pady=5)
    precision_combobox.set("fp16")
    precision_combobox.bind('<<ComboboxSelected>>', lambda event: update_tflops_entry(event, widgets)) # Bind handler
    widgets['precision_combobox'] = precision_combobox
    row_index += 1

    # GPU Selection (New)
    ttk.Label(input_frame, text="GPU:").grid(column=0, row=row_index, sticky=tk.W, padx=5, pady=5)
    gpu_var = tk.StringVar()
    widgets['gpu_var'] = gpu_var
    gpu_names = ["Custom"] + calculator.get_available_gpus()
    gpu_combobox = ttk.Combobox(input_frame, textvariable=gpu_var, values=gpu_names, width=25, state='readonly')
    gpu_combobox.grid(column=1, row=row_index, sticky=(tk.W, tk.E), padx=5, pady=5)
    gpu_combobox.set("Custom")
    gpu_combobox.bind('<<ComboboxSelected>>', lambda event: update_tflops_entry(event, widgets)) # Bind handler
    widgets['gpu_combobox'] = gpu_combobox
    row_index += 1

    # GPU TFLOPS (Now auto-filled)
    ttk.Label(input_frame, text="GPU TFLOPS (@Precision):").grid(column=0, row=row_index, sticky=tk.W, padx=5, pady=5)
    gpu_tflops_entry = ttk.Entry(input_frame, width=20)
    gpu_tflops_entry.insert(0, "312") # Default A100 FP16 TFLOPS
    gpu_tflops_entry.grid(column=1, row=row_index, sticky=(tk.W, tk.E), padx=5, pady=5)
    widgets['gpu_tflops_entry'] = gpu_tflops_entry
    row_index += 1

    # Compute Efficiency
    ttk.Label(input_frame, text="Compute Efficiency (%):").grid(column=0, row=row_index, sticky=tk.W, padx=5, pady=5)
    efficiency_entry = ttk.Entry(input_frame, width=20)
    efficiency_entry.insert(0, "30")
    efficiency_entry.grid(column=1, row=row_index, sticky=(tk.W, tk.E), padx=5, pady=5)
    widgets['efficiency_entry'] = efficiency_entry
    row_index += 1

    # --- Calculate Button ---
    calculate_button = ttk.Button(main_frame, text="Calculate", command=lambda: calculate_requirements(widgets))
    calculate_button.grid(column=0, row=1, pady=10, sticky=tk.W, padx=5)

    # --- Results Area Frame ---
    results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
    results_frame.grid(column=0, row=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
    main_frame.rowconfigure(2, weight=1)
    results_frame.columnconfigure(1, weight=1)

    res_row = 0
    # VRAM Results Section
    ttk.Label(results_frame, text="VRAM:", font=('Segoe UI', 9, 'bold')).grid(column=0, row=res_row, sticky=tk.W, padx=5, pady=(5,2))
    res_row += 1

    ttk.Label(results_frame, text="  Total Required:").grid(column=0, row=res_row, sticky=tk.W, padx=15, pady=1) # Indented
    total_vram_value = ttk.Label(results_frame, text="-", anchor=tk.W)
    total_vram_value.grid(column=1, row=res_row, sticky=(tk.W, tk.E), padx=5, pady=1)
    widgets['total_vram_value'] = total_vram_value
    res_row += 1

    ttk.Label(results_frame, text="  Weights VRAM:").grid(column=0, row=res_row, sticky=tk.W, padx=15, pady=1)
    weights_vram_value = ttk.Label(results_frame, text="-", anchor=tk.W)
    weights_vram_value.grid(column=1, row=res_row, sticky=(tk.W, tk.E), padx=5, pady=1)
    widgets['weights_vram_value'] = weights_vram_value
    res_row += 1

    ttk.Label(results_frame, text="  KV Cache VRAM:").grid(column=0, row=res_row, sticky=tk.W, padx=15, pady=1)
    kv_cache_vram_value = ttk.Label(results_frame, text="-", anchor=tk.W)
    kv_cache_vram_value.grid(column=1, row=res_row, sticky=(tk.W, tk.E), padx=5, pady=1)
    widgets['kv_cache_vram_value'] = kv_cache_vram_value
    res_row += 1

    ttk.Label(results_frame, text="  Activations VRAM:").grid(column=0, row=res_row, sticky=tk.W, padx=15, pady=1)
    activations_vram_value = ttk.Label(results_frame, text="-", anchor=tk.W)
    activations_vram_value.grid(column=1, row=res_row, sticky=(tk.W, tk.E), padx=5, pady=1)
    widgets['activations_vram_value'] = activations_vram_value
    res_row += 1

    ttk.Label(results_frame, text="  System Overhead:").grid(column=0, row=res_row, sticky=tk.W, padx=15, pady=1)
    system_overhead_value = ttk.Label(results_frame, text="-", anchor=tk.W)
    system_overhead_value.grid(column=1, row=res_row, sticky=(tk.W, tk.E), padx=5, pady=1)
    widgets['system_overhead_value'] = system_overhead_value
    res_row += 1

    # FLOPs Section
    ttk.Label(results_frame, text="FLOPs:", font=('Segoe UI', 9, 'bold')).grid(column=0, row=res_row, sticky=tk.W, padx=5, pady=(5,2))
    res_row += 1

    ttk.Label(results_frame, text="  Per Output Token:").grid(column=0, row=res_row, sticky=tk.W, padx=15, pady=1)
    flops_per_token_value = ttk.Label(results_frame, text="-", anchor=tk.W)
    flops_per_token_value.grid(column=1, row=res_row, sticky=(tk.W, tk.E), padx=5, pady=1)
    widgets['flops_per_token_value'] = flops_per_token_value
    res_row += 1

    # Performance Section
    ttk.Label(results_frame, text="Performance:", font=('Segoe UI', 9, 'bold')).grid(column=0, row=res_row, sticky=tk.W, padx=5, pady=(5,2))
    res_row += 1

    ttk.Label(results_frame, text="  Prefill Latency:").grid(column=0, row=res_row, sticky=tk.W, padx=15, pady=1)
    prefill_latency_value = ttk.Label(results_frame, text="-", anchor=tk.W)
    prefill_latency_value.grid(column=1, row=res_row, sticky=(tk.W, tk.E), padx=5, pady=1)
    widgets['prefill_latency_value'] = prefill_latency_value
    res_row += 1

    ttk.Label(results_frame, text="  Token Latency:").grid(column=0, row=res_row, sticky=tk.W, padx=15, pady=1)
    token_latency_value = ttk.Label(results_frame, text="-", anchor=tk.W)
    token_latency_value.grid(column=1, row=res_row, sticky=(tk.W, tk.E), padx=5, pady=1)
    widgets['token_latency_value'] = token_latency_value
    res_row += 1

    ttk.Label(results_frame, text="  Time to First Token:").grid(column=0, row=res_row, sticky=tk.W, padx=15, pady=1)
    ttft_value = ttk.Label(results_frame, text="-", anchor=tk.W)
    ttft_value.grid(column=1, row=res_row, sticky=(tk.W, tk.E), padx=5, pady=1)
    widgets['ttft_value'] = ttft_value
    res_row += 1

    ttk.Label(results_frame, text="  Total Request Time:").grid(column=0, row=res_row, sticky=tk.W, padx=15, pady=1)
    trt_value = ttk.Label(results_frame, text="-", anchor=tk.W)
    trt_value.grid(column=1, row=res_row, sticky=(tk.W, tk.E), padx=5, pady=1)
    widgets['trt_value'] = trt_value
    res_row += 1

    ttk.Label(results_frame, text="  Tokens per Second:").grid(column=0, row=res_row, sticky=tk.W, padx=15, pady=1)
    tokens_per_sec_value = ttk.Label(results_frame, text="-", anchor=tk.W)
    tokens_per_sec_value.grid(column=1, row=res_row, sticky=(tk.W, tk.E), padx=5, pady=1)
    widgets['tokens_per_sec_value'] = tokens_per_sec_value
    res_row += 1

    # Parallelism Section (Using Text Box)
    ttk.Label(results_frame, text="Parallelism Info:", font=('Segoe UI', 9, 'bold')).grid(column=0, row=res_row, sticky=tk.W, padx=5, pady=(5,2))
    res_row += 1

    parallelism_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, height=8, width=40, state=tk.DISABLED)
    parallelism_text.grid(column=0, row=res_row, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)
    widgets['parallelism_text'] = parallelism_text
    res_row += 1

    # --- Status Bar ---
    status_label = ttk.Label(main_frame, text="Status: Ready", relief=tk.SUNKEN, anchor=tk.W)
    status_label.grid(column=0, row=3, sticky=(tk.W, tk.E), padx=5, pady=(5,0))
    widgets['status_label'] = status_label

    # Initialize model selection state
    on_model_select(None, widgets)
    update_tflops_entry(None, widgets) # Initial TFLOPS fill based on default Custom/fp16

    root.mainloop()

if __name__ == "__main__":
    main() 