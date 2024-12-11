def save_previous_outputs_to_file(filename='previous_outputs.txt'):
    # Access the `Out` dictionary that stores outputs of previous cells
    from IPython import get_ipython
    ipython = get_ipython()
    
    # The `Out` dictionary holds all outputs indexed by execution count
    outputs = ipython.user_ns.get('Out', {})
    
    with open(filename, 'w') as file:
        for execution_count, output in outputs.items():
            if output is not None:
                # Write execution count and output to the file
                file.write(f"Cell [{execution_count}]:\n")
                file.write(str(output) + '\n\n')

    print(f"Previous outputs saved to {filename}")

# Run the function to save outputs
save_previous_outputs_to_file()
