from IPython import get_ipython
import json

def extract_existing_outputs(save_to_file='existing_outputs.txt'):
    # Access the kernel's in-memory representation of the notebook
    ipython = get_ipython()
    cells = ipython.history_manager.db  # Accessing history DB (in-memory representation)
    
    extracted_outputs = []
    
    for session_id, line_number, code in ipython.history_manager.get_range():
        try:
            # Skip code, focusing only on outputs
            cell_output = ipython.run_cell(code)._iresult
            if isinstance(cells):
                print("total_outputs")

