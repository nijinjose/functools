from IPython import get_ipython
import json

def capture_previous_cell_output(save_to_file='previous_output.txt'):
    # Get the current IPython shell
    ipython = get_ipython()

    # Access the execution history of the shell
    history = ipython.history_manager
    outputs = []

    # Extract outputs from previous cells in the current session
    for session_id, line_number, source in history.get_range():
        if source:
            try:
                # Capture the output from the executed source
                result = ipython.run_cell(source)._iresult
                if result:
                    outputs.append(str(result))
            except Exception as e:
                outputs.append(f"Error capturing output: {e}")

    # Save outputs to a file
    with open(save_to_file, 'w', encoding='utf-8') as f:
        for output in outputs:
            f.write(output + '\n\n')

    print(f"Output saved to {save_to_file}")

# Run the function
capture_previous_cell_output()
