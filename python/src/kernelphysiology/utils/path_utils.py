'''
Utility functions for path, file and folder related.
'''


import os


def create_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)