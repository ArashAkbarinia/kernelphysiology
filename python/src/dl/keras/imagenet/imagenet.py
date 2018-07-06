'''
Reading the IMAGENET dataset.
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import commons
import logging
import os.path
import tarfile
import tempfile
from collections import OrderedDict
from contextlib import contextmanager

import h5py
import numpy as np
from scipy.io.matlab import loadmat
from six.moves import zip, xrange

import glob
#from fuel.datasets import H5PYDataset

from keras.preprocessing.image import load_img

log = logging.getLogger(__name__)

# path to the directory with the devkit
DEVKIT_DATA_DIR = os.path.join(commons.project_dir, 'data/computervision/ilsvrc/ilsvrc2012/validation')
DEVKIT_META_PATH = os.path.join(DEVKIT_DATA_DIR, 'ILSVRC2012_validation_meta.mat')
DEVKIT_VALID_GROUNDTRUTH_PATH = os.path.join(DEVKIT_DATA_DIR, 'ILSVRC2012_validation_ground_truth.txt')

#TRAIN_IMAGES_TAR = 'ILSVRC2012_img_train.tar'
#VALID_IMAGES_TAR = 'ILSVRC2012_img_val.tar'
#TEST_IMAGES_TAR = 'ILSVRC2012_img_test.tar'


def read_test_labels():
    """Extract dataset metadata required for HDF5 file setup.

    Returns
    -------
    n_train : int
        The number of examples in the training set.
    valid_groundtruth : ndarray, 1-dimensional
        An ndarray containing the validation set groundtruth in terms of
        0-based class indices.
    n_test : int
        The number of examples in the test set
    wnid_map : dict
        A dictionary that maps WordNet IDs to 0-based class indices.

    """
    # Read what's necessary from the development kit.
    synsets, raw_valid_groundtruth = read_devkit()

    # Mapping to take WordNet IDs to our internal 0-999 encoding.
    wnid_map = dict(zip((s.decode('utf8') for s in synsets['WNID']), xrange(1000)))

    # Map the 'ILSVRC2012 ID' to our zero-based ID.
    ilsvrc_id_to_zero_based = dict(zip(synsets['ILSVRC2012_ID'], xrange(len(synsets))))

    # Map the validation set groundtruth to 0-999 labels.
    valid_groundtruth = [ilsvrc_id_to_zero_based[id_] for id_ in raw_valid_groundtruth]

    ntest = 50000
    y_test = [''] * ntest
    for i in valid_groundtruth:
        b = list(wnid_map.keys())[list(wnid_map.values()).index(i)]
        y_test[i] = b


    return y_test
#
#
#def create_splits(n_train, n_valid, n_test):
#    n_total = n_train + n_valid + n_test
#    tuples = {}
#    tuples['train'] = (0, n_train)
#    tuples['valid'] = (n_train, n_train + n_valid)
#    tuples['test'] = (n_train + n_valid, n_total)
#    sources = ['encoded_images', 'targets', 'filenames']
#    return OrderedDict(
#        (split, OrderedDict((source, tuples[split]) for source in sources
#                            if source != 'targets' or split != 'test'))
#        for split in ('train', 'valid', 'test')
#    )
#
#
#@contextmanager
#def create_temp_tar():
#    try:
#        _, temp_tar = tempfile.mkstemp(suffix='.tar')
#        with tarfile.open(temp_tar, mode='w') as tar:
#            tar.addfile(tarfile.TarInfo())
#        yield temp_tar
#    finally:
#        os.remove(temp_tar)
#
#
#def prepare_hdf5_file(hdf5_file, n_train, n_valid, n_test):
#    """Create datasets within a given HDF5 file.
#
#    Parameters
#    ----------
#    hdf5_file : :class:`h5py.File` instance
#        HDF5 file handle to which to write.
#    n_train : int
#        The number of training set examples.
#    n_valid : int
#        The number of validation set examples.
#    n_test : int
#        The number of test set examples.
#
#    """
#    n_total = n_train + n_valid + n_test
#    n_labeled = n_train + n_valid
#    splits = create_splits(n_train, n_valid, n_test)
#    hdf5_file.attrs['split'] = H5PYDataset.create_split_array(splits)
#    vlen_dtype = h5py.special_dtype(vlen=np.dtype('uint8'))
#    hdf5_file.create_dataset('encoded_images', shape=(n_total,), dtype=vlen_dtype)
#    hdf5_file.create_dataset('targets', shape=(n_labeled, 1), dtype=np.int16)
#    hdf5_file.create_dataset('filenames', shape=(n_total, 1), dtype='S32')


def read_devkit():
    """Read relevant information from the development kit archive.

    Returns
    -------
    synsets : ndarray, 1-dimensional, compound dtype
        See :func:`read_metadata_mat_file` for details.
    raw_valid_groundtruth : ndarray, 1-dimensional, int16
        The labels for the ILSVRC2012 validation set,
        distributed with the development kit code.

    """
    synsets = read_metadata_mat_file(DEVKIT_META_PATH)

    # Raw validation data groundtruth, ILSVRC2012 IDs. Confusingly
    # distributed inside the development kit archive.
    raw_valid_groundtruth = np.loadtxt(DEVKIT_VALID_GROUNDTRUTH_PATH, dtype=np.int16)
    return synsets, raw_valid_groundtruth


def read_metadata_mat_file(meta_mat):
    """Read ILSVRC2012 metadata from the distributed MAT file.

    Parameters
    ----------
    meta_mat : str or file-like object
        The filename or file-handle for `meta.mat` from the
        ILSVRC2012 development kit.

    Returns
    -------
    synsets : ndarray, 1-dimensional, compound dtype
        A table containing ILSVRC2012 metadata for the "synonym sets"
        or "synsets" that comprise the classes and superclasses,
        including the following fields:
         * `ILSVRC2012_ID`: the integer ID used in the original
           competition data.
         * `WNID`: A string identifier that uniquely identifies
           a synset in ImageNet and WordNet.
         * `wordnet_height`: The length of the longest path to
           a leaf node in the FULL ImageNet/WordNet hierarchy
           (leaf nodes in the FULL ImageNet/WordNet hierarchy
           have `wordnet_height` 0).
         * `gloss`: A string representation of an English
           textual description of the concept represented by
           this synset.
         * `num_children`: The number of children in the hierarchy
           for this synset.
         * `words`: A string representation, comma separated,
           of different synoym words or phrases for the concept
           represented by this synset.
         * `children`: A vector of `ILSVRC2012_ID`s of children
           of this synset, padded with -1. Note that these refer
           to `ILSVRC2012_ID`s from the original data and *not*
           the zero-based index in the table.
         * `num_train_images`: The number of training images for
           this synset.

    """
    mat = loadmat(meta_mat, squeeze_me=True)
    synsets = mat['synsets']
    new_dtype = np.dtype([
        ('ILSVRC2012_ID', np.int16),
        ('WNID', ('S', max(map(len, synsets['WNID'])))),
        ('wordnet_height', np.int8),
        ('gloss', ('S', max(map(len, synsets['gloss'])))),
        ('num_children', np.int8),
        ('words', ('S', max(map(len, synsets['words'])))),
        ('children', (np.int8, max(synsets['num_children']))),
        ('num_train_images', np.uint16)
    ])
    new_synsets = np.empty(synsets.shape, dtype=new_dtype)
    for attr in ['ILSVRC2012_ID', 'WNID', 'wordnet_height', 'gloss', 'num_children', 'words', 'num_train_images']:
        new_synsets[attr] = synsets[attr]
    children = [np.atleast_1d(ch) for ch in synsets['children']]
    padded_children = [np.concatenate((c, -np.ones(new_dtype['children'].shape[0] - len(c), dtype=np.int16)))
        for c in children
    ]
    new_synsets['children'] = padded_children
    return new_synsets

#def read_train_images(dirname):
#    label_counter = 0
#
#    training_images = []
#    training_labels = []
#    
#    for subdir, dirs, files in os.walk('/data/datasets/imagenet_resized/train/'):
#        for folder in dirs:
#            for folder_subdir, folder_dirs, folder_files in os.walk(os.path.join(subdir, folder)):
#                for file in folder_files:
#                    training_images.append(os.path.join(folder_subdir, file))
#                    training_labels.append(label_counter)
#            label_counter = label_counter + 1


def read_test_images(dirname, rows=224, cols=224, chns=3):
     image_list = sorted(glob.glob(dirname + '*.png'))
     nimages = len(image_list)
     x_test = np.zeros((nimages, rows, cols, chns))
     for i in range(0, nimages):
         current_image = image_list[i]
         img = load_img(current_image, target_size=(rows, cols))
         x_test[i, :, :, :] = img
     return x_test

def load_test_data(dirname):
    # load the test data and labels.
    x_test = read_test_images(dirname)
    y_test = read_test_labels()

    return (x_test, y_test)
