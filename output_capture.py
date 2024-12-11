from IPython import get_ipython
import nbformat

# Get the current IPython shell
ipython = get_ipython()

# Access the current notebook's cell history
shell = ipython.history_manager

# Function to extract outputs from previous runs
def extract_cell_outputs(save_to_file='output_log.txt'):
    outputs = []
    
    # Iterate over the history of cells
    for session_id, line_number, cell_content in shell.get_range():
        if cell_content:
            # Capture the executed cell and its output
            try:
                cell_output = ipython.run_cell(cell_content)._iresult
                if cell_output:
                    outputs.append(str(cell_output))
            except Exception as e:
                outputs.append(f"Error in cell: {e}")
    
    # Write all collected outputs to a file
    with open(save_to_file, 'w') as f:
        for output in outputs:
            f.write(output + '\n\n')

    print(f"Outputs saved to {save_to_file}")

# Run the function
extract_cell_outputs()
