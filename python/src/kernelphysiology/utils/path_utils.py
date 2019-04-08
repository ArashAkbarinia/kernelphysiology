'''
Utility functions for path, file and folder related.
'''


import os


def create_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)


def get_folder_name(folder_path):
    tokens = folder_path.split('/')
    for token in reversed(tokens):
        if token != '':
            return token