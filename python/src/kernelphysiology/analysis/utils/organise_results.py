"""
A collection of functions to organise the output of different experiments.
"""


import os
import glob
import ntpath
import shutil


def arrange_to_network_dirs(input_folder, output_folder, network_names):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # looping through experiments
    for experiment_input in glob.glob(input_folder + '/*/'):
        experiment_name = experiment_input.split('/')[-2]
        print(experiment_name)
        experiment_output = output_folder + '/' + experiment_name + '/'
        if not os.path.exists(experiment_output):
            os.mkdir(experiment_output)
        # going through the output of each network
        for file in glob.glob(experiment_input + '*.csv'):
            network_name = None
            # FIXME: place the name of network at the start ...
            for name in network_names:
                if name in file:
                    network_name = name
                    break
            if network_name is None:
                print(file)
            else:
                network_folder = experiment_output + network_name
                if not os.path.exists(network_folder):
                    os.mkdir(network_folder)
                file_name = ntpath.basename(file)
                test_name = file_name[28:]
                dest = network_folder + '/' + test_name
                shutil.copy(file, dest)
