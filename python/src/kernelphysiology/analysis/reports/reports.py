"""
Reporting stuff!
TODO: should think of how to organise it.
"""

import numpy as np
import glob

from kernelphysiology.utils.path_utils import get_folder_name
from kernelphysiology.analysis.reports.plots import plot_results
from kernelphysiology.analysis.utils.organise_results import ResultSummary
from kernelphysiology.analysis.utils.organise_results import \
    test_value_from_path


def do_one_network(result_dir, which_experiments, original_values, class_inds,
                   which_categories={}, which_networks=None):
    # reading the results for original images
    for experiment_dir in glob.glob(result_dir + '/original/'):
        original_networks = {}

        num_networks = len(glob.glob(experiment_dir + '/*/'))
        for network_dir in sorted(glob.glob(experiment_dir + '/*/')):
            network_name = get_folder_name(network_dir)
            if which_networks is not None and not network_name in which_networks:
                continue
            num_tests = len(glob.glob(network_dir + '/*.csv'))
            original_networks[network_name] = ResultSummary(network_name, 1,
                                                            class_inds)
            for prediction_file in sorted(glob.glob(network_dir + '/*.csv')):
                predictions = np.loadtxt(prediction_file, delimiter=',')
                original_networks[network_name].add_test(predictions, 0)

    summary_out = {}
    for w_ind, experiment_name in enumerate(which_experiments):
        networks = {}

        experiment_dir = result_dir + '/' + experiment_name + '/'
        for network_dir in sorted(glob.glob(experiment_dir + '/*/')):
            network_name = get_folder_name(network_dir)
            if which_networks is not None and not network_name in which_networks:
                continue
            num_tests = len(glob.glob(network_dir + '/*.csv'))
            networks[network_name] = ResultSummary(network_name, num_tests,
                                                   class_inds)
            for prediction_file in sorted(glob.glob(network_dir + '/*.csv')):
                predictions = np.loadtxt(prediction_file, delimiter=',')
                test_name = get_folder_name(prediction_file)
                test_value = test_value_from_path(test_name[:-4])
                networks[network_name].add_test(predictions, test_value)

        (all_xs, all_ys) = plot_results(networks, original_networks,
                                        experiment_name, '_all',
                                        original_values[w_ind])
        for cat_name, cat_val in which_categories.items():
            (_, _) = plot_results(networks, original_networks,
                                  experiment_name, '_' + cat_name,
                                  original_values[w_ind],
                                  cat_val)
        summary_out[experiment_name] = (all_xs, all_ys)
    return summary_out
