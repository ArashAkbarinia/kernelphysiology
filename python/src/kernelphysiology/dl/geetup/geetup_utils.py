"""
Functions to manage the GEETUP dataset.
"""

import numpy as np
import pickle
import logging
import os
import glob

from kernelphysiology.utils.path_utils import create_dir
from kernelphysiology.utils.imutils import max_pixel_ind


def max_pixel_euclidean_distance(a, b):
    a_ind = np.asarray(max_pixel_ind(a))
    b_ind = np.asarray(max_pixel_ind(b))
    return np.linalg.norm(a_ind - b_ind)


def map_point_to_image_size(point, target_size, org_size):
    if target_size[0] == org_size[0] and target_size[1] == org_size[1]:
        return point
    rows = target_size[0]
    cols = target_size[1]
    org_rows = org_size[0]
    org_cols = org_size[1]
    new_p_row = int(round(point[0] * (rows / org_rows)))
    new_p_col = int(round(point[1] * (cols / org_cols)))
    return [new_p_row, new_p_col]


def cleanup_dataset(dataset_dir):
    for subject_dir in sorted(glob.glob(dataset_dir + '/*/')):
        print(subject_dir)
        cleanup_subject(subject_dir)


def create_gt_conds(gts, margin, rows, cols):
    # NOTE: the first dimension of gt goes with rows, and the second with cols

    conds = np.zeros((gts.shape[0], 4))
    conds[:, 0] = np.all(
        [[gts[:, 0] < margin], [gts[:, 1] < margin]], axis=0
    ).squeeze()
    conds[:, 1] = np.all(
        [[gts[:, 0] < margin], [gts[:, 1] > (rows - margin)]], axis=0
    ).squeeze()
    conds[:, 2] = np.all(
        [[gts[:, 0] > (cols - margin)], [gts[:, 1] < margin]], axis=0
    ).squeeze()
    conds[:, 3] = np.all(
        [[gts[:, 0] > (cols - margin)], [gts[:, 1] > (rows - margin)]], axis=0
    ).squeeze()
    conds = np.invert(np.any(conds, axis=1))
    return conds


def cleanup_subject(subject_dir):
    rows = 360
    cols = 640
    margin = 5
    for segment in sorted(glob.glob(subject_dir + '/segments/*/')):
        for video_dir in sorted(glob.glob(segment + '/CutVid_*/')):
            video_list = sorted(glob.glob(video_dir + '/*.jpg'))
            current_num_frames = len(video_list)
            if video_dir[-4:-1] == 'BAD':
                video_ind = video_dir.split('/')[-2].split('_')[-2]
                ignore_video_dir = '%s/Ignore_CutVid_%s/' % (segment, video_ind)
                os.rename(video_dir, ignore_video_dir)
                continue
            video_ind = video_dir.split('/')[-2].split('_')[-1]
            gts = np.loadtxt(segment + '/SUBSAMP_EYETR_' + video_ind + '.txt')
            if gts.shape[0] != current_num_frames:
                logging.info(
                    '%s contains %d frames but %d fixation points' %
                    (video_dir, current_num_frames, gts.shape[0])
                )
                ignore_video_dir = '%s/Ignore_CutVid_%s/' % (segment, video_ind)
                os.rename(video_dir, ignore_video_dir)
                continue

            # ignoring all fixation points that are around corner
            conds = create_gt_conds(gts, margin, rows, cols)
            gts = np.round(gts).astype('int')

            selected_imgs = []
            discarded_imgs = []
            for i in range(current_num_frames):
                im_name = video_list[i].split('/')[-1]
                previous_frame = True
                next_frame = True
                if i != 0:
                    previous_frame = conds[i - 1]
                if i != (current_num_frames - 1):
                    next_frame = conds[i + 1]
                if previous_frame and conds[i] and next_frame:
                    selected_imgs.append([im_name, gts[i, :]])
                else:
                    discarded_imgs.append([im_name, gts[i, :]])
            np.savetxt(
                segment + '/SELECTED_IMGS_' + video_ind + '.txt',
                np.array(selected_imgs), delimiter=',', fmt='%s'
            )
            np.savetxt(
                segment + '/DISCARDED_IMGS_' + video_ind + '.txt',
                np.array(discarded_imgs), delimiter=',', fmt='%s'
            )


def last_valid_frame(selected_gts, frames_gap, sequence_length):
    """
    Given the length of sequence and gap between frames, it computes which is
    the last valid frame to read from a directory.
    """
    selected_frames = np.loadtxt(selected_gts, dtype=str, delimiter=',')
    num_frames = len(selected_frames)
    last_frame = num_frames - (frames_gap * sequence_length)
    return last_frame


def subject_frame_limits(subject_dir, frames_gap=10, sequence_length=9):
    subject_data = []
    for segment in sorted(glob.glob(subject_dir + '/segments/*/')):
        for video_dir in sorted(glob.glob(segment + '/CutVid_*/')):
            video_ind = video_dir.split('/')[-2].split('_')[-1]
            current_video_limit = last_valid_frame(
                segment + '/SELECTED_IMGS_' + video_ind + '.txt',
                frames_gap, sequence_length
            )
            if current_video_limit > 0:
                subject_data.append([segment, video_ind, current_video_limit])
    return subject_data


def dataset_frame_list(dataset_dir, frames_gap=10, sequence_length=9):
    dataset_data = []
    for subject_dir in sorted(glob.glob(dataset_dir + '/*/')):
        subject_data = subject_frame_limits(
            subject_dir, frames_gap=frames_gap, sequence_length=sequence_length
        )
        dataset_data.extend(subject_data)
    return dataset_data


def get_random_video_indices(video_list, num_randoms):
    experiments_inds = []
    previous_experiment = video_list[0][0]
    start_ind = 0
    for i, v in enumerate(video_list):
        current_experiment = v[0]
        if current_experiment != previous_experiment:
            experiments_inds.append([start_ind, i])
            start_ind = i + 1
            previous_experiment = v[0]

    random_videos = []
    for e_ind in experiments_inds:
        rand_ints = set([])
        if abs(e_ind[0] - e_ind[1]) < (num_randoms + 2):
            continue
        while len(rand_ints) < num_randoms:
            rand_ints.add(np.random.randint(e_ind[0], e_ind[1]))
        while len(rand_ints) > 0:
            current_ind = rand_ints.pop()
            random_videos.append(current_ind)
    return random_videos


def create_train_test_sets(video_list, test_experiments=None, num_randoms=7,
                           test_all_subjects_template=None):
    if test_experiments is None:
        test_experiments = [1, 7, 13, 21, 26, 33, 39]

    random_videos = get_random_video_indices(video_list, num_randoms)

    all_sets = {
        'training_both_routes': [],
        'training_route_1': [],
        'training_route_2': [],
        'testing_inter_subjects': [],
        'testing_all_subjects': [],
    }

    for i, v in enumerate(video_list):
        is_for_training = True
        for j in test_experiments:
            experiment_name = 'Part%.3d' % j
            if experiment_name in v[0]:
                all_sets['testing_inter_subjects'].append(v)
                is_for_training = False
                break

        if test_all_subjects_template is None:
            if i in random_videos:
                all_sets['testing_all_subjects'].append(v)
                is_for_training = False
        else:
            for j in test_all_subjects_template:
                if v[0] == j[0] and v[1] == j[1]:
                    all_sets['testing_all_subjects'].append(v)
                    is_for_training = False
                    break

        if is_for_training:
            all_sets['training_both_routes'].append(v)
            if 'Route2' in v[0]:
                all_sets['training_route_2'].append(v)
            else:
                all_sets['training_route_1'].append(v)
    return all_sets


def check_folder_create(output_folder, dataset_dir, frames_gap=10,
                        sequence_length=9):
    create_dir(output_folder)
    dir_name = 'gap_%.3d_seq_%.3d' % (frames_gap, sequence_length)
    output_folder = os.path.join(output_folder, dir_name)
    create_dir(output_folder)
    video_file = output_folder + '/video_list.pickle'
    if os.path.exists(video_file):
        pickle_in = open(video_file, 'rb')
        pickle_info = pickle.load(pickle_in)
        pickle_in.close()
        video_list = pickle_info['video_list']
    else:
        video_list = dataset_frame_list(
            dataset_dir, frames_gap=frames_gap, sequence_length=sequence_length
        )
        pickle_info = {
            'video_list': video_list,
            'sequence_length': sequence_length,
            'frames_gap': frames_gap
        }
        # saving all video list
        pickle_out = open(video_file, 'wb')
        pickle.dump(pickle_info, pickle_out)
        pickle_out.close()
    return video_list, output_folder


def check_train_test_create(output_folder, video_list, frames_gap,
                            sequence_length, test_all_subjects_template=None):
    all_sets = create_train_test_sets(
        video_list, test_all_subjects_template=test_all_subjects_template
    )

    for key, item in all_sets.items():
        pickle_info = {
            'video_list': item,
            'sequence_length': sequence_length,
            'frames_gap': frames_gap
        }
        video_file = output_folder + '/' + key + '.pickle'
        pickle_out = open(video_file, 'wb')
        pickle.dump(pickle_info, pickle_out)
        pickle_out.close()
        print(
            '#videos %d, seq %d, gap %d' %
            (len(item), pickle_info['sequence_length'],
             pickle_info['frames_gap'])
        )


def create_sample_dataset(output_folder, dataset_dir, frames_gap=10,
                          sequence_length=9):
    video_list, output_sub_folder = check_folder_create(
        output_folder, dataset_dir, frames_gap, sequence_length
    )
    test_all_subjects_template = None
    if frames_gap != 10 or sequence_length != 9:
        name_010009 = '/gap_010_seq_009'
        pickle_in = open(
            output_folder + name_010009 + '/testing_all_subjects.pickle', 'rb'
        )
        video_info_010009 = pickle.load(pickle_in)
        test_all_subjects_template = video_info_010009['video_list']
    check_train_test_create(
        output_sub_folder, video_list, frames_gap, sequence_length,
        test_all_subjects_template,
    )
