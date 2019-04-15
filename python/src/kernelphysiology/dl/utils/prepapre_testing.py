"""
A set of common functions used in testing networks across different frameworks.
"""

import os
import numpy as np

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
