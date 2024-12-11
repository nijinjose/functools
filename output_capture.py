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










# Access the output of previous cells in Jupyter Notebook
from IPython.display import display
import sys

# Function to save the outputs to a text file
def save_previous_outputs(save_to_file='output_log.txt'):
    outputs = []
    
    # Get the current IPython shell
    ipython = get_ipython()

    # Iterate through all executed cells
    for cell_id, cell_output in ipython.user_ns.items():
        try:
            if cell_output is not None:
                # Only capture outputs that are not None
                outputs.append(str(cell_output))
        except Exception as e:
            outputs.append(f"Error capturing output for cell {cell_id}: {e}")
    
    # Write outputs to a text file
    with open(save_to_file, 'w') as f:
        for output in outputs:
            f.write(output + '\n\n')

    print(f"Previous outputs saved to {save_to_file}")

# Run the function
save_previous_outputs()

