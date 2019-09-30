"""
Functions to manage the GEETUP dataset.
"""

import numpy as np
import logging
import os
import glob


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


def last_valid_frame(video_dir, frames_gap=10, sequence_length=9):
    """
    Given the length of sequence and gap between frames, it computes which is
    the last valid frame to read from a directory.
    """
    num_frames = len(glob.glob((video_dir + '/*.jpg')))
    last_frame = num_frames - (frames_gap * sequence_length)
    return last_frame


def subject_frame_limits(subject_dir, frames_gap=10, sequence_length=9):
    subject_data = []
    for segment in sorted(glob.glob(subject_dir + '/segments/*/')):
        for video_dir in sorted(glob.glob(segment + '/CutVid_*/')):
            video_ind = video_dir.split('/')[-2].split('_')[-1]
            current_video_limit = last_valid_frame(
                video_dir + '/selected/', frames_gap, sequence_length
            )
            if current_video_limit > 0:
                subject_data.append([segment, video_ind, current_video_limit])
    return subject_data
