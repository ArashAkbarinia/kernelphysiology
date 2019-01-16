
# coding: utf-8

# In[1]:


import cv2
import keras
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
import argparse
import sys
import os


# In[2]:


parser = argparse.ArgumentParser(description='Merging two networks.')
parser.add_argument('--network_src', type=str, help='The source network.')
parser.add_argument('--network_des', type=str, help='The destination network.')
parser.add_argument('--output_folder', type=str, help='The folder to write the results')
parser.add_argument('--gpus', nargs='+', type=int, default=[0], help='List of GPUs to be used (default: [0])')
parser.add_argument('--thresholds', nargs='+', type=float, default=[0.75, 0.90, 0.95, 1.0], help='List of thresholds (default: [0.75, 0.90, 0.95, 1.0])')

args = parser.parse_args(sys.argv[1:])

os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join(str(e) for e in args.gpus)


# In[3]:


def compute_metric(all_weights1, all_weights2, metric='pearsonr'):
    all_metric = []
    for weights1, weights2 in zip(all_weights1, all_weights2):
        kernel_metric = np.zeros((weights1.shape[3], 1))
        for i in range(weights1.shape[3]):
            kernel1 = weights1[:,:,:,i]
            kernel2 = weights2[:,:,:,i]
            if metric == 'pearsonr':
                (r, p) = pearsonr(kernel1.flatten(), kernel2.flatten())
            #r = mutual_info_score(kernel1.flatten(), kernel2.flatten())
            #r = normalized_mutual_info_score(kernel1.flatten(), kernel2.flatten())
            #r = np.mean(np.abs(kernel1.flatten() - kernel2.flatten()))
            kernel_metric[i, 0] = r
        all_metric.append(kernel_metric)
    return all_metric


def get_kernels_metric_threshold(all_metric, layer_names, threshold):
    metric_summery = []
    for i in range(len(layer_names)):
        conditioned_list = all_metric[i] <= threshold
        passed_inds = [j for j, x in enumerate(conditioned_list) if x]
        metric_summery.append(passed_inds)
    return metric_summery


# getting the weights, names, their difference for convolutional layers and depthwise conv
# keras.layers.core.Dense
def merge_weights(network_des, network_src, which_layers, which_kernels):
    for i, layer_name in enumerate(which_layers):
        layer_des = network_des.get_layer(layer_name).get_weights()
        layer_src = network_src.get_layer(layer_name).get_weights()
        if type(network_des.get_layer(layer_name)) is keras.layers.convolutional.Conv2D and which_kernels:
            for j in which_kernels[i]:
                layer_des[0][:,:,:,j] = layer_src[0][:,:,:,j]
                layer_des[1][j] = layer_src[1][j]
        elif type(network_des.get_layer(layer_name)) is keras.layers.normalization.BatchNormalization and which_kernels:
            for j in which_kernels[i]:
                for n in range(4):
                    layer_des[n][j] = layer_src[n][j]
        else:
            layer_des = layer_src
        network_des.get_layer(layer_name).set_weights(layer_des)
    return network_des


# getting the weights, names, their difference for convolutional layers and depthwise conv
# keras.layers.core.Dense
def get_weights(network,
                which_layers=[keras.layers.convolutional.Conv2D, keras.layers.DepthwiseConv2D]):
    all_weights = []
    layer_names = []
    for i, layer in enumerate(network.layers):
        if type(layer) in which_layers:
            layer_names.append(layer.name)

            weights = layer.get_weights()

            all_weights.append(weights[0])
    return (all_weights, layer_names)


# In[4]:

layer_type = [keras.layers.convolutional.Conv2D]

for th_value in args.thresholds:
    network_src = keras.models.load_model(args.network_src)
    network_des = keras.models.load_model(args.network_des)

    (all_weights_src, layer_names) = get_weights(network_src, which_layers=layer_type)
    (all_weights_des, layer_names) = get_weights(network_des, which_layers=layer_type)

    corr_metric_src_des = compute_metric(all_weights_src, all_weights_des, metric='pearsonr')
    threshold_src_des = get_kernels_metric_threshold(corr_metric_src_des, layer_names, th_value)

    which_layers = []
    which_kernels = []
    for i in range(len(layer_names)):
        kernels = threshold_src_des[i]
        if kernels:
            which_layers.append(layer_names[i])
            which_kernels.append(kernels)
    for i in range(len(which_layers)):
        if 'branch' in which_layers[i]:
            which_layers.append(which_layers[i].replace('res', 'bn'))
            which_kernels.append(which_kernels[i])
        else:
            which_layers.append('bn_' + which_layers[i][0:-3])
            which_kernels.append(which_kernels[i])

    network_des = merge_weights(network_des, network_src, which_layers, which_kernels)
    network_des.save(args.output_folder + '/restnet50_corr%d_merged.h5' % (th_value * 100))

