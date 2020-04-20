"""
Utility functions for path, file and folder related.
"""

import os
import pickle
import glob

IMG_EXTENSIONS = [
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp'
]


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


def _read_extension(root, extension):
    img_paths = []
    img_paths.extend(
        sorted(glob.glob(root + '/*' + extension))
    )
    # with upper case
    img_paths.extend(
        sorted(glob.glob(root + '/*' + extension.upper()))
    )
    return img_paths


def image_in_folder(root, extensions=None):
    if extensions is None:
        extensions = IMG_EXTENSIONS

    img_paths = []
    # reading all extensions
    for extension in extensions:
        img_paths.extend(_read_extension(root, extension))

    return img_paths
