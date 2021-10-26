from pathlib import Path
import os
import subprocess
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors import CellExecutionError


# make sure we're in the right working directory
os.chdir(Path(__file__).parent.resolve())
# find all jupyter notebooks in the data folder
files = sorted(Path(f'../data').rglob("*.ipynb"))

for file in files:
    print(file)

    with open(file) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    #os.system(f'jupyter nbconvert --execute --to notebook --inplace {f}')
    #ep.preprocess(nb)
    try:
        out = ep.preprocess(nb)
    except CellExecutionError:
        out = None
        msg = 'Error executing the notebook \n\n'
        print(msg)
        raise
    finally:
        with open(file, mode='w', encoding='utf-8') as f:
            nbformat.write(nb, f)
    #run = subprocess.run(f'jupyter nbconvert --execute --to notebook --inplace {f}', stdout=subprocess.DEVNULL)
    
