"""
A set of common functions used in testing networks across different frameworks.
"""

import os
import numpy as np
import glob
import pickle

from kernelphysiology.utils.path_utils import create_dir


def prepare_saving_dir(experiment_name, network, dataset, manipulation_type):
    create_dir(experiment_name)
    dataset_dir = os.path.join(experiment_name, dataset)
    create_dir(dataset_dir)
    manipulation_dir = os.path.join(dataset_dir, manipulation_type)
    create_dir(manipulation_dir)
    network_dir = os.path.join(manipulation_dir, network)
    create_dir(network_dir)
    return network_dir


def _prepare_saving_file(experiment_name, network, dataset, manipulation_type,
                         manipulation_value, extension, chunk=''):
    network_dir = prepare_saving_dir(
        experiment_name, network, dataset, manipulation_type
    )
    file_name = '%s_%s_%s%s.%s' % (
        network, manipulation_type, str(manipulation_value), chunk, extension)
    output_file = os.path.join(network_dir, file_name)
    return output_file


def save_predictions(predictions, experiment_name, network, dataset,
                     manipulation_type, manipulation_value):
    output_file = _prepare_saving_file(
        experiment_name, network, dataset, manipulation_type,
        manipulation_value, extension='csv'
    )
    np.savetxt(output_file, predictions, delimiter=',', fmt='%i')


def save_segmentation_results(predictions, experiment_name, network, dataset,
                              manipulation_type, manipulation_value):
    pred_log = predictions.get_log_dict()
    tobesaved = []
    header = ''
    for key, val in pred_log.items():
        tobesaved.append(str(val))
        if header != '':
            header += ','
        header = header + key
    output_file = _prepare_saving_file(
        experiment_name, network, dataset, manipulation_type,
        manipulation_value, extension='csv'
    )
    np.savetxt(
        output_file, np.array([tobesaved]),
        delimiter=';', header=header, fmt='%s'
    )


def save_activation(activations, experiment_name, network, dataset,
                    manipulation_type, manipulation_value):
    j = 1
    gap = 5000
    for start_i in range(0, len(activations), gap):
        output_file = _prepare_saving_file(
            experiment_name, network, dataset, manipulation_type,
            manipulation_value, extension='pickle', chunk='_chunk%.3d' % j
        )
        pickle_out = open(output_file, 'wb')
        end_i = start_i + gap
        pickle.dump(activations[start_i:end_i], pickle_out)
        pickle_out.close()
        j += 1


def prepare_networks_testting(network_arg, network_chromaticity='trichromat'):
    if os.path.isdir(network_arg):
        dirname = network_arg
        network_files = sorted(glob.glob(dirname + '*.h5'))
        network_names = []
        network_chromaticities = [network_chromaticity] * len(network_files)
    elif os.path.isfile(network_arg):
        network_files = []
        network_chromaticities = []
        network_names = []
        with open(network_arg) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                tokens = line.strip().split(',')
                network_files.append(tokens[0])
                # if colour transformation is not provided, use default
                if len(tokens) > 1:
                    network_chromaticities.append(tokens[1])
                else:
                    network_chromaticities.append(network_chromaticity)
                # if network name is provided, genrate one
                if len(tokens) > 2:
                    network_names.append(tokens[2])
                else:
                    network_names.append('network_%03d' % i)
    else:
        network_files = [network_arg.lower()]
        network_names = [network_arg.lower()]
        network_chromaticities = [network_chromaticity]

    return network_files, network_names, network_chromaticities
