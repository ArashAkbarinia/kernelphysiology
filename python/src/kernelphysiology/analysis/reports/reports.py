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
                   which_categories=None, which_networks=None):
    # reading the results for original images
    if which_categories is None:
        which_categories = {}
    experiment_dir = result_dir + '/original/'
    original_results = read_experiment(experiment_dir, which_networks,
                                       class_inds)
    experiment_dir = result_dir + '/original_rgb/'
    original_rgb_results = read_experiment(experiment_dir, which_networks,
                                           class_inds)

    summary_out = {}
    for w_ind, experiment_name in enumerate(which_experiments):
        experiment_dir = result_dir + '/' + experiment_name + '/'
        other_results = read_experiment(experiment_dir, which_networks,
                                        class_inds)

        original_plot_results = original_results
        if experiment_name in ['reduce_chromaticity',
                               'red_green',
                               'yellow_blue']:
            original_plot_results = original_rgb_results
        (all_xs, all_ys) = plot_results(other_results, original_plot_results,
                                        experiment_name, '_all',
                                        original_values[w_ind])
        for cat_name, cat_val in which_categories.items():
            (_, _) = plot_results(other_results, original_plot_results,
                                  experiment_name, '_' + cat_name,
                                  original_values[w_ind],
                                  cat_val)
        summary_out[experiment_name] = (all_xs, all_ys)
    return summary_out


def read_experiment(experiment_dir, which_networks, class_inds):
    network_results = {}

    for network_dir in sorted(glob.glob(experiment_dir + '/*/')):
        network_name = get_folder_name(network_dir)
        if which_networks is not None and network_name not in which_networks:
            continue
        num_tests = len(glob.glob(network_dir + '/*.csv'))
        test_value = 0
        network_results[network_name] = ResultSummary(network_name, num_tests,
                                                      class_inds)
        for prediction_file in sorted(glob.glob(network_dir + '/*.csv')):
            predictions = np.loadtxt(prediction_file, delimiter=',')
            if num_tests > 1:
                test_name = get_folder_name(prediction_file)
                test_value = test_value_from_path(test_name[:-4])
            network_results[network_name].add_test(predictions, test_value)
    return network_results
