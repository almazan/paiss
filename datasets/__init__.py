from .oxford import *

def create( dataset_cmd ):
    ''' Create a dataset from a string.

    dataset_cmd (str):
        Command to execute.
        ex: "ImageList('path/to/list.txt')"

    Returns:
        instanciated dataset.
    '''
    if '(' not in dataset_cmd:
        dataset_cmd += "()"

    return eval(dataset_cmd)
