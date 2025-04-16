// Format VRAM to display in GB with 2 decimal places
function formatVRAM(vramBytes) {
    if (vramBytes === null || vramBytes === undefined || Number.isNaN(parseFloat(vramBytes))) {
        return "0.00 GB";
    }
    return (vramBytes / (1024 * 1024 * 1024)).toFixed(2) + " GB";
}

// Handle form submission
$('#calculatorForm').submit(function(e) {
    e.preventDefault();
    
    // Show loading indicator
    $('#loadingOverlay').show();
    $('#results').hide();
    
    // Clear previous results
    $('#resultsContainer').html('');
    
    // Get form data
    const formData = {
        calculation_type: 'model',
        model_name: $('#model-preset').val() || 'custom',
        hidden_dim: parseInt($('#hidden-dim').val()) || 0,
        ff_dim: parseInt($('#ff-dim').val()) || 0,
        num_layers: parseInt($('#num-layers').val()) || 0,
        vocab_size: parseInt($('#vocab-size').val()) || 0,
        seq_length: parseInt($('#seq-len').val()) || 2048,
        batch_size: parseInt($('#batch-size').val()) || 1,
        precision: $('#precision').val() || 'fp16',
        gpu: $('#gpu-model').val() || '',
        efficiency_factor: parseFloat($('#efficiency').val()) || 0.3
    };
    
    // Add parallelism info if needed
    const parallelismStrategy = $('#parallelism-strategy').val();
    if (parallelismStrategy !== 'none') {
        formData.parallelism = {
            strategy: parallelismStrategy,
            tp_size: parseInt($('#tp-size').val()) || 1,
            pp_size: parseInt($('#pp-size').val()) || 1,
            num_gpus: parseInt($('#num-gpus').val()) || 1
        };
    }
    
    // Submit calculation request
    $.ajax({
        url: '/api/calculate',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(formData),
        success: function(response) {
            // Hide loading indicator and show results
            $('#loadingOverlay').hide();
            $('#results').show();
            
            // Display results
            displayResults(response);
            
            // Log the calculation to history
            addToHistory(response);
        },
        error: function(error) {
            // Hide loading indicator
            $('#loadingOverlay').hide();
            
            // Show error message
            $('#resultsContainer').html(`
                <div class="alert alert-danger">
                    <h4>Error</h4>
                    <p>${error.responseJSON?.error || 'An unknown error occurred'}</p>
                </div>
            `);
        }
    });
});

// Display calculation results
function displayResults(data) {
    // Handle missing data gracefully
    if (!data) {
        $('#resultsContainer').html('<div class="alert alert-danger">No data received from calculator</div>');
        return;
    }
    
    if (data.error) {
        $('#resultsContainer').html(`
            <div class="alert alert-danger">
                <h4>Calculator Error</h4>
                <p>${data.error}</p>
            </div>
        `);
        return;
    }
    
    const vram = data.vram || {};
    const flops = data.flops || {};
    const performance = data.performance || {};
    const parallelism = data.parallelism || { strategy: 'none', num_gpus: 1 };
    const overheads = data.overheads_used || {};

    // Determine if we're using parallelism
    const isParallel = parallelism.strategy !== 'none' && parallelism.num_gpus > 1;
    
    // Create the main results container
    let html = `
        <div class="card mb-4">
            <div class="card-header bg-primary text-white">
                <h5 class="mb-0">LLM Runtime Requirements</h5>
            </div>
            <div class="card-body">
                <h5 class="card-title">${data.model_name} (${data.parameters_billions?.toFixed(2) || '?'} billion parameters)</h5>
                
                <div class="row">
                    <!-- VRAM Requirements -->
                    <div class="col-md-6">
                        <div class="card mb-3">
                            <div class="card-header bg-info text-white">
                                <h6 class="mb-0">VRAM Requirements</h6>
                            </div>
                            <div class="card-body">
                                <table class="table table-sm">
                                    <thead>
                                        <tr>
                                            <th>Component</th>
                                            <th>Base</th>
                                            <th>With Overhead</th>
                                            ${isParallel ? '<th>Per GPU</th>' : ''}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr>
                                            <td>Model Weights</td>
                                            <td>${formatVRAM(vram.model_base)}</td>
                                            <td>${formatVRAM(vram.model_with_overhead)}</td>
                                            ${isParallel ? `<td>${formatVRAM(vram.model_per_gpu_with_overhead)}</td>` : ''}
                                        </tr>
                                        <tr>
                                            <td>KV Cache</td>
                                            <td>${formatVRAM(vram.kv_cache_base)}</td>
                                            <td>${formatVRAM(vram.kv_cache_with_overhead)}</td>
                                            ${isParallel ? `<td>${formatVRAM(vram.kv_cache_per_gpu_with_overhead)}</td>` : ''}
                                        </tr>
                                        <tr>
                                            <td>Activations</td>
                                            <td>${formatVRAM(vram.activations_base)}</td>
                                            <td>${formatVRAM(vram.activations_with_overhead)}</td>
                                            ${isParallel ? `<td>${formatVRAM(vram.activations_per_gpu_with_overhead)}</td>` : ''}
                                        </tr>
                                        <tr class="table-info font-weight-bold">
                                            <td>Total</td>
                                            <td>${formatVRAM(vram.total_base)}</td>
                                            <td>${formatVRAM(vram.total_with_component_overhead)}</td>
                                            ${isParallel ? `<td>${formatVRAM(vram.total_per_gpu_with_component_overhead)}</td>` : ''}
                                        </tr>
                                    </tbody>
                                </table>
                                <div class="mt-2">
                                    <div class="alert alert-primary">
                                        <strong>System-wide VRAM requirement:</strong> ${formatVRAM(vram.total_system_wide)}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Computation Requirements -->
                    <div class="col-md-6">
                        <div class="card mb-3">
                            <div class="card-header bg-warning text-dark">
                                <h6 class="mb-0">Computation Requirements</h6>
                            </div>
                            <div class="card-body">
                                <h6>FLOPs</h6>
                                <table class="table table-sm">
                                    <tbody>
                                        <tr>
                                            <td>Attention FLOPs</td>
                                            <td>${(flops.attention / 1e9).toFixed(2)} GFLOPs</td>
                                        </tr>
                                        <tr>
                                            <td>Feedforward FLOPs</td>
                                            <td>${(flops.feedforward / 1e9).toFixed(2)} GFLOPs</td>
                                        </tr>
                                        <tr>
                                            <td>Prefill Total FLOPs</td>
                                            <td>${((flops.prefill_total || 0) / 1e12).toFixed(2)} TFLOPs</td>
                                        </tr>
                                        <tr>
                                            <td>Per-token FLOPs</td>
                                            <td>${(flops.per_token / 1e9).toFixed(2)} GFLOPs</td>
                                        </tr>
                                    </tbody>
                                </table>
                                
                                <h6 class="mt-3">Performance Estimates</h6>
                                <table class="table table-sm">
                                    <tbody>
                                        <tr>
                                            <td>Prefill Latency</td>
                                            <td>${performance.prefill_latency?.toFixed(2) || '?'} seconds</td>
                                        </tr>
                                        <tr>
                                            <td>Token Generation Latency</td>
                                            <td>${performance.token_latency?.toFixed(2) || '?'} seconds</td>
                                        </tr>
                                        <tr>
                                            <td>Tokens Per Second</td>
                                            <td>${performance.tokens_per_second?.toFixed(2) || '?'} tokens/sec</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Parallelism & GPU Info -->
                <div class="row">
                    <div class="col-md-6">
                        <div class="card mb-3">
                            <div class="card-header bg-success text-white">
                                <h6 class="mb-0">Parallelism & GPU Settings</h6>
                            </div>
                            <div class="card-body">
                                <table class="table table-sm">
                                    <tbody>
                                        <tr>
                                            <td>Strategy</td>
                                            <td>${parallelism.strategy || 'None'}</td>
                                        </tr>
                                        <tr>
                                            <td>Number of GPUs</td>
                                            <td>${parallelism.num_gpus || 1}</td>
                                        </tr>
                                        <tr>
                                            <td>Tensor Parallel Size</td>
                                            <td>${parallelism.tp_size || 1}</td>
                                        </tr>
                                        <tr>
                                            <td>Pipeline Parallel Size</td>
                                            <td>${parallelism.pp_size || 1}</td>
                                        </tr>
                                        <tr>
                                            <td>Effective TFLOPS</td>
                                            <td>${parallelism.effective_tflops?.toFixed(2) || '?'} TFLOPS</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-6">
                        <div class="card mb-3">
                            <div class="card-header bg-secondary text-white">
                                <h6 class="mb-0">Overhead Factors</h6>
                            </div>
                            <div class="card-body">
                                <table class="table table-sm">
                                    <tbody>
                                        <tr>
                                            <td>Model Weights</td>
                                            <td>${overheads.weights?.toFixed(2) || '1.05'}</td>
                                        </tr>
                                        <tr>
                                            <td>KV Cache</td>
                                            <td>${overheads.kv_cache?.toFixed(2) || '1.05'}</td>
                                        </tr>
                                        <tr>
                                            <td>Activations</td>
                                            <td>${overheads.activations?.toFixed(2) || '1.10'}</td>
                                        </tr>
                                        <tr>
                                            <td>System</td>
                                            <td>${overheads.system?.toFixed(2) || '1.05'}</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Calculation Steps -->
                <div class="row mt-2">
                    <div class="col-12">
                        <button class="btn btn-outline-secondary btn-sm" type="button" data-toggle="collapse" data-target="#calculationSteps">
                            Show Calculation Steps
                        </button>
                        <div class="collapse mt-2" id="calculationSteps">
                            <div class="card">
                                <div class="card-header bg-light">
                                    <h6 class="mb-0">Calculation Steps</h6>
                                </div>
                                <div class="card-body">
                                    <pre class="bg-light p-3">${formatCalculationSteps(data.history || [])}</pre>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Add the HTML to the results container
    $('#resultsContainer').html(html);
}

// Format calculation steps for display
function formatCalculationSteps(steps) {
    if (!steps || !Array.isArray(steps) || steps.length === 0) {
        return "No calculation steps available";
    }
    
    return steps.join('\n');
} 