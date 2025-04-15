from src.advanced_calculator.main import AdvancedCalculator

def test_calculator():
    # Create calculator instance
    calculator = AdvancedCalculator()
    
    # Call the analyze_model_by_name method that we fixed
    try:
        result = calculator.analyze_model_by_name(
            model_name="llama3-8b",
            sequence_length=1000,
            batch_size=1,
            precision="fp16",
            gpu_tflops=989,
            efficiency_factor=0.3
        )
        
        # Print the relevant parts of the result to verify the fix
        print("Analysis successful!")
        print("\nParallelism Info:")
        print(result.get("parallelism", "Missing parallelism field"))
        
        print("\nOverheads Used:")
        print(result.get("overheads_used", "Missing overheads_used field"))
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")

if __name__ == "__main__":
    test_calculator() 