
# coding: utf-8

# In[1]:


import cv2
import keras
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
import os
import math
import sys
import argparse


# In[2]:


parser = argparse.ArgumentParser(description='Comparing all networks ina round robin fashion.')
parser.add_argument('--network_paths', type=str, help='Which network to be used')
parser.add_argument('--output_folder', type=str, help='The folder to write the results')
parser.add_argument('--gpus', nargs='+', type=int, default=[0], help='List of GPUs to be used (default: [0])')

args = parser.parse_args(sys.argv[1:])

os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join(str(e) for e in args.gpus)
network_paths = args.network_paths

networks = []
paths = []
with open(network_paths) as f:
    lines = f.readlines()
    for line in lines:
        tokens = line.strip().split(',')
        if os.path.isfile(tokens[0]):
            paths.append(tokens[0])
            networks.append(0)
        else:
            print(tokens[0])


# In[9]:


def compare_networks(network_i, network_j, which_layers):
    ij_compare = []
    for l in which_layers:
        layer_i = network_i[l]
        layer_j = network_j[l]
        weights_i = layer_i.get_weights()[0]
        weights_j = layer_j.get_weights()[0]

        ijw_compare = []
        # convolutional layer
        if len(weights_i.shape) > 2:
            r_max = 0
            takens = np.zeros((weights_i.shape[3]))
            for w_i in range(weights_i.shape[3]):
                kernel_i = weights_i[:,:,:,w_i]
                for w_j in range(weights_j.shape[3]):
                    if w_j in takens:
                        continue
                    kernel_j = weights_j[:,:,:,w_j]
                    (r, _) = pearsonr(kernel_i.flatten(), kernel_j.flatten())
                    if math.isnan(r):
                        r = 0
                    if r > r_max:
                        r_max = r
                        takens[w_i] = w_j
                ijw_compare.append(r_max)
        else:
            (r, _) = pearsonr(weights_i.flatten(), weights_j.flatten())
            ijw_compare.append(r)

        ijw_compare = np.array(ijw_compare)
        ij_compare.append(ijw_compare.mean())
    return ij_compare


# In[8]:


num_networks = len(paths)
num_layers = 0
network_i = keras.models.load_model(paths[0])
which_layers = []
for l, layer in enumerate(network_i.layers):
    invert_op = getattr(layer, "get_weights", None)
    if callable(invert_op) and layer.get_weights():
        which_layers.append(l)
        num_layers += 1
network_comparison = np.zeros((num_networks, num_networks)) - 1
network_comparison_layers = np.zeros((num_networks, num_networks, num_layers)) - 1


# In[0]:
for k in range(num_layers):
    if os.path.isfile(args.output_folder + '/network_comparison%02d.csv' % k):
        network_comparison_layers[:,:,k] = np.loadtxt(args.output_folder + '/network_comparison%02d.csv' % k, delimiter=',')
if os.path.isfile(args.output_folder + '/network_comparison_kernel_max.csv'):
    network_comparison = np.loadtxt(args.output_folder + '/network_comparison_kernel_max.csv', delimiter=',')


# In[10]:


chunk_size = 20
for c1 in range(0, num_networks, chunk_size):
    networks = [0] * num_networks
    for c2 in range(c1, c1+chunk_size):
        print('loading model %d' % c2)
        networks[c2] = keras.models.load_model(paths[c2])
    for i in range(c1, c1+chunk_size-1):
        network_i = networks[i].layers
        for j in range(c1+1, c1+chunk_size):
            if network_comparison[i, j] == -1:
                print('Processing networks %d %d' % (i, j))
                network_j = networks[j].layers
                ij_compare = compare_networks(network_i, network_j, which_layers)
                ij_compare = np.array(ij_compare)
                network_comparison_layers[i, j, :] = ij_compare
                network_comparison_layers[j, i, :] = network_comparison_layers[i, j, :]
                network_comparison[i, j] = ij_compare.mean()
                network_comparison[j, i] = network_comparison[i, j]
                for k in range(num_layers):
                    np.savetxt(args.output_folder + '/network_comparison%02d.csv' % k, network_comparison_layers[:,:,k], delimiter=',')
                np.savetxt(args.output_folder + '/network_comparison_kernel_max.csv', network_comparison, delimiter=',')

