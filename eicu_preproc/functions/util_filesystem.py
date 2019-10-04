""" 
File system utilities
"""

import os
import os.path

def delete_if_exist(path):
    if os.path.exists(path):
        os.remove(path)

def create_dir_if_not_exist(path, recursive=False):
    if not os.path.exists(path):
        if recursive:
            os.makedirs(path)
        else:
            os.mkdir(path)

