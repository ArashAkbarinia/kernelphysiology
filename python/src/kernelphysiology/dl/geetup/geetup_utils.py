"""
Functions to manage the GEETUP dataset.
"""

import numpy as np
import logging
import os
import glob

from kernelphysiology.utils.path_utils import create_dir
from kernelphysiology.utils.path_utils import write_pickle
from kernelphysiology.utils.path_utils import read_pickle
from kernelphysiology.utils.imutils import max_pixel_ind
from kernelphysiology.utils.controls import isint


def get_video_clips_inds(geetup_info):
    last_folder = None
    video_clips = []
    for j in range(0, geetup_info.__len__()):
        f_path, f_gt = geetup_info.__getitem__(j)
        prt_num = f_path[-1].split('/')[-7]
        seg_num = f_path[-1].split('/')[-4]
        vid_num = f_path[-1].split('/')[-2]
        current_folder = '%s_%s_%s' % (prt_num, seg_num, vid_num)
        if last_folder is None:
            print('Starting %s' % current_folder)
            last_folder = current_folder
            start_ind = j
        elif current_folder != last_folder:
            video_clips.append([start_ind, end_ind])
            print('Starting %s' % current_folder)
            start_ind = j
            last_folder = current_folder
        end_ind = j + 1
    video_clips.append([start_ind, end_ind])
    return video_clips


def parse_gt_line(gt):
    gt = gt.replace('[', '').replace(']', '').split(' ')
    gt = [int(i) for i in gt if isint(i)]
    # in the file it's stored as x and y, rather than rows and cols
    gt = gt[::-1]
    return gt


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


def _random_video_indices(video_list, percent, exclude_list=None):
    if exclude_list is None:
        exclude_list = []
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
    for i, e_ind in enumerate(experiments_inds):
        current_experiment_inds = [*range(e_ind[0], e_ind[1])]
        num_elemnts = round(len(current_experiment_inds) * percent)
        for ri in exclude_list:
            if ri in current_experiment_inds:
                current_experiment_inds.remove(ri)
        np.random.shuffle(current_experiment_inds)
        random_videos.extend(current_experiment_inds[:num_elemnts])

    return random_videos


def get_inidividual_name(video_path, num_persons):
    for j in range(1, num_persons):
        experiment_name = 'Part%.3d' % j
        if experiment_name in video_path:
            return experiment_name
    return None


def _is_in_video_paths(video, videos_set):
    for j in videos_set:
        experiment_name = 'Part%.3d' % j
        if experiment_name in video:
            return True
    return False


def _is_in_subj_clips(clip, clips_set):
    for j in clips_set:
        if clip[0] == j[0] and clip[1] == j[1]:
            return True
    return False


def _is_in_clips_set(video_ind, subjs_set, exp_name, random_set, video):
    if subjs_set is None:
        if video_ind in random_set:
            return True
    else:
        if _is_in_subj_clips(video, subjs_set[exp_name]):
            return True
    return False


def create_train_test_sets(video_list, test_subjs=None, val_subjs=None,
                           nontrain=0.40, test_clips=None, val_clips=None):
    if test_subjs is None:
        test_subjs = [13, 20, 23, 33, 39]
    if val_subjs is None:
        val_subjs = [1, 7, 21, 26, 44]

    # TODO: right now essentially test_clips and val_clips are connected to
    #  eachother, the exclude_list passed should support dictionaries
    nontrain /= 2
    random_test = None
    random_val = None
    if test_clips is None:
        random_test = _random_video_indices(video_list, nontrain, random_val)
    if val_clips is None:
        random_val = _random_video_indices(video_list, nontrain, random_test)

    all_sets = {
        'training_both_routes': [],
        'training_route_1': [],
        'training_route_2': [],
        'validation_both_routes': [],
        'validation_route_1': [],
        'validation_route_2': [],
        'testing_both_routes': [],
        'testing_route_1': [],
        'testing_route_2': [],
    }

    num_persons = 45
    # creating individualised data
    for v_ind in range(1, num_persons):
        exp_name = 'Part%.3d' % v_ind
        all_sets[exp_name] = {'training': [], 'validation': [], 'testing': []}

    for v_ind, vid in enumerate(video_list):
        is_for_training = True
        if _is_in_video_paths(vid[0], test_subjs):
            all_sets['testing_both_routes'].append(vid)
            if 'Route2' in vid[0]:
                all_sets['testing_route_2'].append(vid)
            else:
                all_sets['testing_route_1'].append(vid)
            is_for_training = False
        elif _is_in_video_paths(vid[0], val_subjs):
            all_sets['validation_both_routes'].append(vid)
            if 'Route2' in vid[0]:
                all_sets['validation_route_2'].append(vid)
            else:
                all_sets['validation_route_1'].append(vid)
            is_for_training = False

        exp_name = get_inidividual_name(vid[0], num_persons)
        if _is_in_clips_set(v_ind, test_clips, exp_name, random_test, vid):
            all_sets[exp_name]['testing'].append(vid)
            is_for_training = False
        elif _is_in_clips_set(v_ind, val_clips, exp_name, random_val, vid):
            all_sets[exp_name]['validation'].append(vid)
            is_for_training = False

        if is_for_training:
            all_sets['training_both_routes'].append(vid)
            all_sets[exp_name]['training'].append(vid)
            if 'Route2' in vid[0]:
                all_sets['training_route_2'].append(vid)
            else:
                all_sets['training_route_1'].append(vid)
    return all_sets


def check_folder_create(out_folder, dataset_dir, frames_gap=10, seq_length=9):
    create_dir(out_folder)
    dir_name = 'gap_%.3d_seq_%.3d' % (frames_gap, seq_length)
    out_folder = os.path.join(out_folder, dir_name)
    create_dir(out_folder)
    video_file = out_folder + '/video_list.pickle'
    if os.path.exists(video_file):
        video_info = read_pickle(video_file)
    else:
        video_list = dataset_frame_list(
            dataset_dir, frames_gap=frames_gap, sequence_length=seq_length
        )
        video_info = {
            'video_list': video_list,
            'sequence_length': seq_length,
            'frames_gap': frames_gap
        }
        video_info = extract_base_path_dic(video_info, dataset_dir, prefix='')
        # saving all video list
        write_pickle(video_file, video_info)
    return video_info, out_folder


def _save_video_dic(key, item, out_folder, **kwargs):
    create_dir(out_folder)
    video_info = {'video_list': item}
    for kwarg_key, kwarg_item in kwargs.items():
        video_info[kwarg_key] = kwarg_item
    video_file = out_folder + '/' + key + '.pickle'
    write_pickle(video_file, video_info)


def _save_all_items(data, out_folder, **kwargs):
    for key, item in data.items():
        if isinstance(item, dict):
            _save_all_items(item, os.path.join(out_folder, key), **kwargs)
        else:
            _save_video_dic(key, item, out_folder, **kwargs)


def check_train_test_create(out_folder, video_info, test_clips=None,
                            val_clips=None):
    all_sets = create_train_test_sets(
        video_info['video_list'], test_clips=test_clips, val_clips=val_clips
    )
    del video_info['video_list']
    _save_all_items(all_sets, out_folder, **video_info)


def _read_all_subjects_pickle(in_folder, file_type):
    all_clips = {}
    for pickle_file in sorted(glob.glob(in_folder + '/Part*/')):
        video_info_010009 = read_pickle(pickle_file + file_type + '.pickle')
        part_name = pickle_file.replace(in_folder, '').replace('/', '')
        all_clips[part_name] = video_info_010009['video_list']
    return all_clips


def create_sample_dataset(out_folder, dataset_dir, frames_gap=10, seq_length=9):
    video_info, out_subfolder = check_folder_create(
        out_folder, dataset_dir, frames_gap, seq_length
    )
    test_clips = None
    val_clips = None
    if frames_gap != 10 or seq_length != 9:
        path_010009 = os.path.join(out_folder, 'gap_010_seq_009')
        test_clips = _read_all_subjects_pickle(path_010009, 'testing')
        val_clips = _read_all_subjects_pickle(path_010009, 'validation')
    check_train_test_create(
        out_subfolder, video_info, test_clips, val_clips
    )


def change_base_path_recursive(in_folder, old, new, out_folder, prefix=''):
    create_dir(out_folder)
    for current_in in sorted(glob.glob(in_folder + '/*/')):
        print(current_in)
        current_out = current_in.replace(in_folder, out_folder)
        create_dir(current_out)
        for in_file in sorted(glob.glob(current_in + '/*.pickle')):
            out_file = in_file.replace(in_folder, out_folder)
            change_base_path(in_file, old, new, out_file, prefix)


def change_base_path(in_file, old, new, out_file, prefix=''):
    data = read_pickle(in_file)

    # changing the base path
    new_base = data['base_path_img']
    new_base = new_base.replace(old, new)
    data['base_path_img'] = new_base

    new_base = data['base_path_txt']
    new_base = new_base.replace(old, new)
    data['base_path_txt'] = new_base

    # changing the prefix
    data['prefix'] = prefix

    write_pickle(out_file, data)


# TODO: make all other recursive folders like this
def change_pickles_recursive(in_folder, out_folder, **kwargs):
    change_pickles_dir(in_folder, out_folder, **kwargs)
    for current_in in sorted(glob.glob(in_folder + '/*/')):
        print(current_in)
        current_out = current_in.replace(in_folder, out_folder)
        change_pickles_recursive(current_in, current_out, **kwargs)


def change_pickles_dir(in_folder, out_folder, **kwargs):
    create_dir(out_folder)
    for in_file in sorted(glob.glob(in_folder + '/*.pickle')):
        out_file = in_file.replace(in_folder, out_folder)
        change_pickles(in_file, out_file, **kwargs)


def change_pickles(in_file, out_file, **kwargs):
    data = read_pickle(in_file)

    for key, item in kwargs.items():
        data[key] = item

    write_pickle(out_file, data)


def remove_segment_video_list_recursive(in_folder, segment):
    for folder in sorted(glob.glob(in_folder + '/*/')):
        print(folder)
        for in_file in sorted(glob.glob(folder + '/*.pickle')):
            remove_segment_video_list(in_file, segment)


def remove_segment_video_list(in_file, segment):
    data = read_pickle(in_file)

    for i in range(len(data['video_list'])):
        new_path = data['video_list'][i][0]
        new_path = new_path.replace(segment, '')
        data['video_list'][i][0] = new_path

    write_pickle(in_file, data)


def extract_base_path_recursive(in_folder, base_path, prefix=''):
    for folder in sorted(glob.glob(in_folder + '/*/')):
        print(folder)
        for in_file in sorted(glob.glob(folder + '/*.pickle')):
            extract_base_path(in_file, base_path, prefix)


def extract_base_path(in_file, base_path, prefix=''):
    data = read_pickle(in_file)
    data = extract_base_path_dic(data, base_path, prefix)
    write_pickle(in_file, data)


def extract_base_path_dic(data, base_path, prefix=''):
    # adding the base path
    data['base_path_img'] = base_path
    data['base_path_txt'] = base_path

    # adding the prefix
    data['prefix'] = prefix

    for i in range(len(data['video_list'])):
        new_path = data['video_list'][i][0]
        new_path = new_path.replace(base_path, '')
        data['video_list'][i][0] = new_path
    return data
