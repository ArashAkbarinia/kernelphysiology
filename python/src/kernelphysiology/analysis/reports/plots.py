"""
Plotting stuff!
TODO: should think of how to organise it.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_results(networks, original_networks, experiment_name, category_name,
                 original_value, cat_inds=None):
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    all_xs = None
    all_ys = []
    for ((key, value), (key_original, value_original)) in \
            zip(networks.items(), original_networks.items()):
        xs = np.concatenate((value.test_values, [[original_value]]), axis=1)
        if cat_inds is None:
            non_original_res = value.top1_accuracy
            original_res = value_original.top1_accuracy
        else:
            non_original_res = value.classes_top1_accuracy[
                               cat_inds.astype('uint'), :].mean(axis=0)
            non_original_res = np.expand_dims(non_original_res, 0)
            original_res = value_original.classes_top1_accuracy[
                           cat_inds.astype('uint'), :].mean(axis=0)
            original_res = np.expand_dims(original_res, 0)
        ys = np.concatenate((non_original_res, original_res), axis=1)
        sorted_inds = np.argsort(xs)
        if all_xs is None:
            all_xs = xs[0, sorted_inds]
        all_ys.append(ys[0, sorted_inds])
        ax.plot(*xs[0, sorted_inds], *ys[0, sorted_inds], marker='o')
    ax.legend(networks.keys(), loc='right', bbox_to_anchor=(1.45, 0.8))
    ax.set_title(experiment_name + category_name)
    ax.set_ylim([0, 1])
    ax.set_ylabel('Top 1 Accuracy')
    ax.set_xlabel(experiment_name)
    all_ys = np.array(all_ys).squeeze()
    ax.axhline(y=all_ys.max(), linestyle='--', color='black')
    plt.show()
    return (all_xs, all_ys)
