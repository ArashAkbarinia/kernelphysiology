'''
Analysing the activity of kernels.
'''


import os
import numpy as np
import commons
import time
import datetime
import argparse

import tensorflow as tf
import keras
from keras.utils import multi_gpu_model





if __name__ == "__main__":
    start_stamp = time.time()
    start_time = datetime.datetime.fromtimestamp(start_stamp).strftime('%Y-%m-%d_%H_%M_%S')
    print('Starting at: ' + start_time)

    parser = argparse.ArgumentParser(description='Analysing the kernels of a network.')
    parser.add_argument(dest='dataset', type=str, help='Which dataset to be used')
    parser.add_argument(dest='network_name', type=str, help='Which network to be used')

    args = parser.parse_args()



    finish_stamp = time.time()
    finish_time = datetime.datetime.fromtimestamp(finish_stamp).strftime('%Y-%m-%d_%H-%M-%S')
    duration_time = (finish_stamp - start_stamp) / 60
    print('Finishing at: %s - Duration %.2f minutes.' % (finish_time, duration_time))