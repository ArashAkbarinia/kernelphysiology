"""
Utility functions for path, file and folder related.
"""

import os
import pickle


def create_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)


def get_folder_name(folder_path):
    tokens = folder_path.split('/')
    for token in reversed(tokens):
        if token != '':
            return token


def read_pickle(in_file):
    pickle_in = open(in_file, 'rb')
    data = pickle.load(pickle_in)
    pickle_in.close()
    return data


def write_pickle(out_file, data):
    pickle_out = open(out_file, 'wb')
    pickle.dump(data, pickle_out)
    pickle_out.close()
