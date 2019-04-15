"""
A set of common functions used in testing networks across different frameworks.
"""

import os
import numpy as np
import glob

from kernelphysiology.utils.path_utils import create_dir


def save_predictions(predictions, experiment_name, network, dataset,
                     manipulation_type, manipulation_value):
    create_dir(experiment_name)
    dataset_dir = os.path.join(experiment_name, dataset)
    create_dir(dataset_dir)
    manipulation_dir = os.path.join(dataset_dir, manipulation_type)
    create_dir(manipulation_dir)
    network_dir = os.path.join(manipulation_dir, network)
    create_dir(network_dir)
    file_name = '%s_%s_%s.csv' % (
        network, manipulation_type, str(manipulation_value))
    output_file = os.path.join(network_dir, file_name)
    np.savetxt(output_file, predictions, delimiter=',', fmt='%i')


def test_prominent_prepares(network_arg, preprocessing=None):
    if os.path.isdir(network_arg):
        dirname = network_arg
        networks = sorted(glob.glob(dirname + '*.h5'))
        network_names = []
        preprocessings = [preprocessing] * len(networks)
    elif os.path.isfile(network_arg):
        networks = []
        preprocessings = []
        network_names = []
        with open(network_arg) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                tokens = line.strip().split(',')
                networks.append(tokens[0])
                if len(tokens) > 1:
                    preprocessings.append(tokens[1])
                else:
                    preprocessings.append(preprocessing)
                if len(tokens) > 2:
                    network_names.append(tokens[2])
                else:
                    network_names.append('network_%03d' % i)
    else:
        networks = [network_arg.lower()]
        network_names = [network_arg.lower()]
        # choosing the preprocessing function
        if preprocessing is None:
            preprocessing = network_arg.lower()
        preprocessings = [preprocessing]

    return networks, network_names, preprocessings
