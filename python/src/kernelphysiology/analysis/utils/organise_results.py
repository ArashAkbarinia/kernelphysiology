"""
A collection of functions to organise the output of different experiments.
"""


import numpy as np
import os
import glob
import ntpath
import shutil

from kernelphysiology.utils import controls


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


def test_value_from_path(test_name):
    # TODO: better to put more specific conventions
    tokens = test_name.split('_')
    for token in reversed(tokens):
        if controls.isfloat(token):
            return token


class ResultSummary():
    test_index = 0

    def __init__(self, name, num_tests, class_inds):
        num_classes = class_inds.shape[0]
        self.name = name
        self.top1_accuracy = np.zeros((1, num_tests))
        self.top5_accuracy = np.zeros((1, num_tests))
        self.classes_top1_accuracy = np.zeros((num_classes, num_tests))
        self.classes_top5_accuracy = np.zeros((num_classes, num_tests))
        self.test_values = np.zeros((1, num_tests))
        self.num_classes = num_classes
        self.class_inds = class_inds

    def add_test(self, predictions, test_value):
        self.test_values[0, self.test_index] = test_value
        self.top1_accuracy[0, self.test_index] = predictions[:, 0].mean()
        self.top5_accuracy[0, self.test_index] = predictions[:, 1].mean()
        for i in range(self.num_classes):
            si = int(self.class_inds[i, 0])
            ei = int(self.class_inds[i, 1])
            self.classes_top1_accuracy[i, self.test_index] = \
                predictions[si:ei, 0].mean()
            self.classes_top5_accuracy[i, self.test_index] = \
                predictions[si:ei, 1].mean()
        self.test_index += 1
