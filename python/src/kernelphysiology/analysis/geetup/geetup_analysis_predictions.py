#!/usr/bin/env python
# coding: utf-8

# # Analysis of Gaze Video data using segmentation labels

# In[1]:


import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import pearsonr
from statistics import mean

# ## Labels and keys definitions

# In[2]:


keys = ["Participant",  # The participant identifier (integer) 1 to 44
        "Route",  # The corresponding route (integer) 1 or 2
        "Video",
        # The different videos per participant (integer) it depends on the participant
        "Frame",  # The considered video frame (string)
        "Gaze_label",
        # The segmentation label of the gaze point (integer) from -1 to 33
        "Pixels_per_label"
        # Number of pixels in the frame corresponding to each segmentation label (narray 1x34)
        ]

from collections import namedtuple
from kernelphysiology.utils.path_utils import write_pickle

# --------------------------------------------------------------------------------
# Definitions
# --------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class

    'id',  # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # evaluation server.

    'trainId',
    # Feel free to modify these IDs as suitable for your method. Then create
    # ground truth images with train IDs, using the tools provided in the
    # 'preparation' folder. However, make sure to validate or submit results
    # to our evaluation server using the regular IDs above!
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!

    'category',  # The name of the category that this label belongs to

    'categoryId',  # The ID of this category. Used to create ground truth images
    # on category level.

    'hasInstances',
    # Whether this label distinguishes between single instances or not

    'ignoreInEval',
    # Whether pixels having this class as ground truth label are ignored
    # during evaluations or not

    'color',  # The color of this label
])

labels = [
    Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('rect. border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
    Label('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
    Label('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    Label('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    Label('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
    Label('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
    Label('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    Label('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    Label('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    Label('guard rail', 14, 255, 'construction', 2, False, True,
          (180, 165, 180)),
    Label('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    Label('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    Label('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    Label('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
    Label('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    Label('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    Label('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    Label('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    Label('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    Label('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    Label('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    Label('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    Label('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    Label('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    Label('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    Label('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    Label('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    Label('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    Label('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
]

cityscapes_id2TrainId = {
    label.id: label.trainId for label in labels
}

allcolor = [label.color for label in labels]
allnames = [label.name for label in labels]
TrainId2id = {label.trainId: label.id for label in labels}
id2TrainId = {label.id: label.trainId for label in labels}
TrainId2labels = {label.trainId: label.name for label in labels}

# Legend with all the labels
train_labels = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                31, 32, 33]  # , -1
blank_image = np.zeros((500, 200, 3), np.uint8)

count = 0
for i in range(0, 500, 50):
    for j in range(0, 200, 100):
        if count < len(train_labels):
            blank_image[i:i + 50, j:j + 100] = allcolor[train_labels[count]]
        else:
            blank_image[i:i + 50, j:j + 100] = allcolor[-1]
        count += 1

    # ## Required Functions

# In[68]:

from joblib import Parallel
from joblib import delayed


def process_one_image(all_gazes, folder_path, im_name, key_p, route, key_ff):
    folder = folder_path + im_name + 'png'
    img_seg = cv2.imread(folder, 0)
    # The gaze coordinates
    all_aux_dicts = []
    for gaze in all_gazes:
        aux_dict = {
            'Participant': int(key_p[-3:]),
            'Route': int(route),
            'Video': int((key_ff.split('_'))[1]),
            'Frame': im_name + 'png',
            'Gaze_label': cityscapes_id2TrainId[
                int(img_seg[gaze[0], gaze[1]])]
        }
        all_aux_dicts.append(aux_dict)
    return all_aux_dicts


def get_folder_path(f_segmentation, key_r, key_f, key_ff):
    folder = f_segmentation + key_r + '/segments/' + key_f + '/' + key_ff + '/' + '/pred_mask/'
    return folder


def get_gazes(all_videos, key_p, key_r, key_f, key_ff, i):
    all_gazes = []
    for video in all_videos:
        all_images = video[key_p, key_r, key_f, key_ff]
        all_gazes.append(all_images[i][2])
    return all_gazes


def create_arrayfromPredictions_one(all_videos, f_segmentation):
    all_videos_array = []
    for video in all_videos:
        all_videos_array.append([])

    for key_p, val_p in all_videos[0].items():
        print('*** PROCESSING ', key_p)
        for key_r, val_r in val_p.items():

            route = 1
            if 'Route2' in key_r:
                route = 2

            for key_f, val_f in val_r.items():
                for key_ff, val_ff in val_f.items():
                    folder_path = get_folder_path(
                        f_segmentation, key_r, key_f, key_ff
                    )
                    for i, im_data in enumerate(val_ff):
                        im_name = im_data[0][:-3]
                        gazes = get_gazes(
                            all_videos, key_p, key_r, key_f, key_ff, i
                        )
                        all_aux_dicts = process_one_image(
                            gazes, folder_path, im_name, key_p, route, key_ff
                        )
                        for j, aux_dict in enumerate(all_aux_dicts):
                            all_videos_array[j].append(aux_dict)

    return all_videos_array


def create_arrayfromPredictions(all_videos, f_segmentation):
    all_videos_array = []

    for key_p in all_videos.keys():
        print(key_p)
        for key_r in all_videos[key_p].keys():

            route = 1
            if 'Route2' in key_r:
                route = 2

            for key_f in all_videos[key_p][key_r].keys():

                for key_ff in all_videos[key_p][key_r][key_f].keys():
                    for i in all_videos[key_p][key_r][key_f][key_ff]:
                        folder = f_segmentation + key_r + '/segments/' + key_f + '/' + key_ff + '/' + '/pred_mask/' + \
                                 i[0][:-3] + 'png'
                        img_seg = cv2.imread(folder, 0)
                        # The gaze coordinates
                        gaze = i[2]
                        aux_dict = {
                            'Participant': int(key_p[-3:]),
                            'Route': int(route),
                            'Video': int((key_ff.split('_'))[1]),
                            'Frame': i[0][:-3] + 'png',
                            'Gaze_label': cityscapes_id2TrainId[
                                int(img_seg[gaze[0], gaze[1]])]
                        }
                        all_videos_array.append(aux_dict)

    return all_videos_array


def read_pickle(in_file):
    pickle_in = open(in_file, 'rb')
    data = pickle.load(pickle_in)
    pickle_in.close()
    return data


# participants_folders = []
# for part_dir in sorted(glob.glob(f_segmentation + '/*/')):
#     participants_folders.append(part_dir)


# def infopickle2array(all_videos, id2TrainId):
#     all_videos_array = []
#
#     for key_p in all_videos.keys():
#         for key_r in all_videos[key_p].keys():
#             if not all_videos[key_p][key_r]:
#                 continue
#             for key_f in all_videos[key_p][key_r].keys():
#
#                 for i in all_videos[key_p][key_r][key_f]:
#                     # print(key_p, key_r, key_f, i)
#                     matching_part = [s for s in participants_folders if
#                                      key_p in s]
#                     if key_r == '2':
#                         matching_all = [s for s in matching_part if
#                                         'Route2' in s]
#                     else:
#                         matching_all = [s for s in matching_part if
#                                         not 'Route2' in s]
#
#                     folder = matching_all[
#                                  0] + 'segments/1/' + key_f + '/pred_mask/' + i[
#                                                                                   0][
#                                                                               :-3] + 'png'
#                     img_seg = cv2.imread(folder, 0)
#                     # The gaze coordinates
#                     gaze = i[2]
#                     aux_dict = {'Participant': int(key_p[-3:]),
#                                 'Route': int(key_r),
#                                 'Video': int((key_f.split('_'))[1]),
#                                 'Frame': i[0][:-3] + 'png',
#                                 'Gaze_label': id2TrainId[
#                                     int(img_seg[gaze[0], gaze[1]])]}
#                     all_videos_array.append(aux_dict)
#
#     return all_videos_array


# Returns frequency gaze (label) with respect a given key (e.g. participant, route)   
def return_query_frequency_gaze(dictionary, key, key_value, train_labels,
                                label='Gaze_label'):
    f_query = np.zeros((len(train_labels)))
    query = [x[label] for x in dictionary if x[key] == key_value]

    # Frequency for each label    
    unique, counts = np.unique(query, return_counts=True, axis=0)
    # print(unique)
    count_query = np.asarray((unique, counts), dtype=np.float32).T
    count_query[:, 1] = count_query[:, 1] / (count_query.sum(axis=0))[1]

    count = 0
    for i in unique:
        f_query[i] = count_query[count, 1]
        count += 1

    return f_query  # count_query[:, 1]


# Returns all frequency labels with respect a given key (e.g. participant, route)
def return_query_frequency_labels(dictionary, key, key_value,
                                  label='Pixels_per_label', train_labels=[]):
    query = [x[label] for x in dictionary if x[key] == key_value]

    route = query[0].replace("[", "")
    route = route.replace("]", "")

    route = np.fromstring(route, dtype=int, sep=',')

    # Frequency for each label   
    for i in range(len(query) - 1):
        aux = query[i + 1].replace("[", "")
        aux = aux.replace("]", "")
        route += np.fromstring(aux, dtype=int, sep=',')

    count = 0
    route_trainlabels = []

    if not train_labels:
        route_trainlabels = route
    else:
        for i in range(len(train_labels)):
            route_trainlabels.append(route[train_labels[i]])
            count += 1

    route_trainlabels = np.asarray(route_trainlabels, dtype=np.float64)
    route_trainlabels /= route_trainlabels.sum()

    return route_trainlabels


# Assign the colors for the segmentation trainId labels
def assign_colors2labels(labels):
    v_col = []
    for i in range(len(labels)):
        v_col.append(np.array(allcolor[TrainId2id[int(labels[i])]]) / 255.)

    return v_col


# def do_gazeObjectCategory_participants(all_videos, train_labels):
#     query_participant = []
#
#     for i in number_participants:
#         aux = return_query_frequency_gaze(all_videos, 'Participant', i,
#                                           train_labels, 'Gaze_label')
#         query_participant.append(aux)
#
#     return query_participant


def do_pearsonGazeObjectCategory_participants(query_participant,
                                              number_participants):
    pear_mat = np.zeros((len(number_participants), len(number_participants)))
    pear_mat.fill(np.nan)

    for i in range(len(number_participants) - 1):
        g_i = query_participant[i].copy()
        for j in range(i + 1, len(number_participants)):
            g_j = query_participant[j].copy()
            corr, _ = pearsonr(g_i.flatten(), g_j.flatten())
            pear_mat[i, j] = corr
            pear_mat[j, i] = corr

    return pear_mat


def do_changeGazeObjectCategory_participants(all_videos, number_participants,
                                             train_labels):
    m_participants = []
    for i in number_participants:  # defined in 2.Compute query gaze label for all the participants

        m = np.zeros((len(train_labels), len(train_labels)))
        query = [x['Gaze_label'] for x in all_videos if x['Participant'] == i]

        for j in range(len(query) - 1):
            if int(query[j]) != int(query[j + 1]):
                m[int(query[j])][int(query[j + 1])] += 1

        # Normalize by the sum in each row
        row_sum = m.sum(axis=1)
        ind = np.where(row_sum == 0)[0]
        row_sum[ind] = 1
        m = (m.T / row_sum).T
        m_participants.append(m)

    return m_participants


def do_pearsonChangeGazeObjectCategory_participants(query_participants,
                                                    number_participants):
    pear_mat = np.ones((len(number_participants), len(number_participants)))
    pear_mat.fill(np.nan)
    for i in range(len(number_participants)):
        m_i = query_participants[i].copy()
        for j in range(i + 1, len(number_participants)):
            m_j = query_participants[j].copy()
            corr, p_value = pearsonr(m_i.flatten(), m_j.flatten())
            pear_mat[i, j] = corr
            pear_mat[j, i] = corr

    return pear_mat


# def do_allObjectCategory_participants(all_videos, train_labels):
#     query_participant_alllabels = []
#
#     for i in number_participants:
#         aux = return_query_frequency_labels(all_videos, 'Participant', i,
#                                             'Pixels_per_label', train_labels)
#         query_participant_alllabels.append(aux)
#     return query_participant_alllabels


def do_pearsonAllObjectCategory_participants(query_participant,
                                             number_participants):
    pear_mat = np.zeros((len(number_participants), len(number_participants)))
    pear_mat.fill(np.nan)
    for i in range(len(number_participants) - 1):
        g_i = query_participant[i].copy()
        for j in range(i + 1, len(number_participants)):
            g_j = query_participant[j].copy()
            corr, _ = pearsonr(g_i.flatten(), g_j.flatten())
            pear_mat[i, j] = corr
            pear_mat[j, i] = corr
    return pear_mat


# ## Required plots 

# In[4]:


def plot_bar_frequency(x, query, v_col, x_label, y_label, xticklabel, title,
                       x_lim=[-1, 20], y_lim=[0, .4], tuple_size=(16, 8)):
    fig = plt.figure(figsize=tuple_size)
    ax = fig.add_subplot(121)
    plt.bar(x, query, color=v_col)
    plt.xlabel(x_label, fontsize=15)
    plt.ylabel(y_label, fontsize=15)

    # Show all ticks
    ax.set_xticks(np.arange(len(x)))
    # Label them with the respective list entries
    ax.set_xticklabels(xticklabel)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.xticks(fontsize=15, rotation=45)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_title(title, fontsize=17)


def plot_scatter(x_data, y_data, v_col, x_label, y_label, title, x_lim=[0., .4],
                 y_lim=[0, .4], tuple_size=(16, 8)):
    fig = plt.figure(figsize=tuple_size)
    ax = fig.add_subplot(121)
    plt.scatter(x_data, y_data, color=v_col, s=1000 * [300])
    plt.xlabel(x_label, fontsize=15)
    plt.ylabel(y_label, fontsize=15)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    plt.title(title, fontsize=17)


def plot_confusion_matrix(conf_mat, number_participants, title, rot=0,
                          tuple_size=(10, 10)):
    fig, ax = plt.subplots(figsize=tuple_size)
    im = ax.imshow(conf_mat)

    # Show all ticks
    ax.set_xticks(np.arange(len(conf_mat)))
    ax.set_yticks(np.arange(len(conf_mat)))
    # Label them with the respective list entries
    ax.set_xticklabels(number_participants)
    ax.set_yticklabels(number_participants)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=rot, ha="right",
             rotation_mode="anchor")
    plt.xticks(fontsize=15, rotation=rot)
    plt.yticks(fontsize=15)

    # Create colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)

    plt.colorbar(im, cax=cax)

    ax.set_title(title, fontsize=17)
    fig.tight_layout()


def autopct_per2(pct):
    return ('%.1f%%' % pct) if pct > 2 else ''


def plot_pie_chart(query, v_col, title, tuple_size=(10, 10)):
    plt.figure(figsize=tuple_size)
    ax = plt.subplot(121)

    _, _, autotexts = ax.pie(query, shadow=True, colors=v_col,
                             autopct=autopct_per2, pctdistance=.8)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(40)
        autotext.set_weight('bold')

    # draw circle
    centre_circle = plt.Circle((0, 0), 0.30, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.title(title, fontsize=17)
    ax.axis('equal')
    plt.tight_layout()


# ## Load data 

# In[ ]:


f_segmentation = '/mnt/data/arash_data/geetup/cityscapes_best_results/'

# In[ ]:


import sys
import glob

in_folder = sys.argv[1]
all_folders = []
all_videos = []
for folder_name in sorted(glob.glob(in_folder + '/Part*/')):
    all_folders.append(folder_name)
    all_videos.append(read_pickle(folder_name + 'all_validation.pickle'))
all_videos_array = create_arrayfromPredictions_one(all_videos, f_segmentation)
for i, video_array in enumerate(all_videos_array):
    current_folder = all_folders[i]
    write_pickle(current_folder + 'all_validation_array.pickle', video_array)
