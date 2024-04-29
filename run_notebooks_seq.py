import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MyExecutePreprocessor(ExecutePreprocessor):
    def preprocess_cell(self, cell, resources, cell_index):
        """
        Override to capture and print outputs directly to the console.
        """
        outputs = super().preprocess_cell(cell, resources, cell_index)
        if 'outputs' in cell:
            for output in cell.outputs:
                if output.output_type == 'stream':
                    print(output.text, end='')
                elif output.output_type == 'error':
                    print("Error: ", output.ename, output.evalue)
                elif output.output_type == 'display_data' and 'image/png' in output.data:
                    print("[Image data]")
                elif output.output_type == 'execute_result' and 'text/plain' in output.data:
                    print(output.data['text/plain'])
        return outputs

def run_notebook(path):
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)
    
    ep = MyExecutePreprocessor(timeout=600, kernel_name='python3')
    
    try:
        # Process the notebook
        ep.preprocess(nb)

        # Optionally save the processed notebook to a new file
        output_path = f"output_{path}"
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        logging.info(f"Successfully ran and saved {path} to {output_path}")

    except Exception as e:
        logging.error(f"Error running {path}: {e}")

notebooks = ['experiments_new_pre.ipynb', 'experiments_old_snr.ipynb','experiments_oldest_pre.ipynb']
for notebook in notebooks:
    run_notebook(notebook)

