"""
A collection of informative functions about miscellaneous subjects.
"""


import numpy as np
import glob

from kernelphysiology import commons
from kernelphysiology.utils.path_utils import get_folder_name


def selected_imagenet_categories(
        cat_folder=commons.project_dir +
                   '/data/computervision/ilsvrc/ilsvrc2012/raw-data/categories/'
):
    selected_categories = {}

    for cat_file in glob.glob(cat_folder + '/*.txt'):
        cat_name = get_folder_name(cat_file[:-4])
        selected_categories[cat_name] = np.loadtxt(cat_file)
    return selected_categories


def imagenet_category_inds():
    num_samples = 50
    num_classes = 1000
    num_images = num_samples * num_classes
    class_inds = np.zeros((num_classes, 2))
    for j, i in enumerate(range(0, num_images, num_samples)):
        class_inds[j, :] = [i, i + num_samples]
    return class_inds
