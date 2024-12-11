from IPython import get_ipython

def extract_existing_outputs_to_file(save_to_file='existing_outputs.txt'):
    """
    Extract outputs from previously executed cells in a Jupyter Notebook
    and save them to a file.
    """
    ipython = get_ipython()
    outputs = []
    
    # Access the Out dictionary to get all stored outputs
    out_dict = ipython.user_ns.get('Out', {})
    
    with open(save_to_file, 'w', encoding='utf-8') as file:
        for execution_count, output in out_dict.items():
            if output is not None:
                # Write each output to the file
                file.write(f"Cell [{execution_count}]:\n")
                file.write(str(output) + '\n\n')
    
    print(f"All outputs saved to {save_to_file}")

# Execute the function to save outputs
extract_existing_outputs_to_file()
