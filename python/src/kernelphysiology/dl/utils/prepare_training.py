"""
Common functionality of training a network under different frameworks.
"""

import os

from kernelphysiology import commons

from kernelphysiology.utils.path_utils import create_dir


def prepare_output_directories(dataset_name, network_name, optimiser,
                               load_weights, experiment_name, framework):
    # preparing directories
    data_folder_path = os.path.join(commons.python_root, 'data')
    create_dir(data_folder_path)
    network_folder_path = os.path.join(data_folder_path, 'nets')
    create_dir(network_folder_path)
    framework_folder_path = os.path.join(network_folder_path, framework)
    create_dir(framework_folder_path)

    # organise the dataset according to their parents
    if 'wcs' in dataset_name:
        dataset_parent = 'wcs'
    else:
        dataset_parent = ''.join([i for i in dataset_name if not i.isdigit()])
    dataset_parent_path = os.path.join(
        framework_folder_path, '%s' % dataset_parent
    )

    create_dir(dataset_parent_path)
    dataset_child_path = os.path.join(dataset_parent_path, dataset_name)
    create_dir(dataset_child_path)
    network_parent_path = os.path.join(dataset_child_path, network_name)
    create_dir(network_parent_path)
    network_dir = os.path.join(network_parent_path, optimiser)
    create_dir(network_dir)
    if load_weights is not None and load_weights is True:
        f_s_dir = os.path.join(network_dir, 'fine_tune')
    else:
        f_s_dir = os.path.join(network_dir, 'scratch')
    create_dir(f_s_dir)
    save_dir = os.path.join(f_s_dir, experiment_name)
    create_dir(save_dir)
    return save_dir
