"""
Collection of functions to analyse the geetup related data.
"""

import os
import glob
import numpy as np

import cv2

from kernelphysiology.datasets import cityscapes_labels
from kernelphysiology.dl.geetup.geetup_utils import parse_gt_line
from kernelphysiology.utils.path_utils import create_dir

cityscapes_id2color = {
    label.id: label.color for label in cityscapes_labels.labels
}


def _cityscape_folder(im_list, dir_path, id2color=None):
    if id2color is None:
        id2color = cityscapes_id2color

    all_results = []
    # going through all the images of the list
    for frame_num in range(im_list.shape[0]):
        # The segmentation image
        img_seg = cv2.imread(
            os.path.join(dir_path, im_list[frame_num][0]), cv2.IMREAD_GRAYSCALE
        )

        # The gaze coordinates
        gaze = parse_gt_line(im_list[frame_num][1])

        pix_per_label = []
        for key in id2color:
            pix_per_label.append(np.count_nonzero(img_seg == key))

        all_results.append(
            [im_list[frame_num][0], img_seg[gaze[0], gaze[1]], pix_per_label]
        )
    return all_results


def report_cityscape(input_folder, out_dir=None, out_suffix='',
                     file_exp='SELECTED_IMGS_*.txt'):
    if out_dir is None:
        out_dir = input_folder
    for part_dir in sorted(glob.glob(input_folder + '/*/')):
        save_part = part_dir.split('/')[-2]
        create_dir(os.path.join(out_dir, save_part))
        save_part_segment = save_part + '/segments/'
        create_dir(os.path.join(out_dir, save_part_segment))
        for video_dir in sorted(glob.glob(part_dir + '/segments/*/')):
            save_segment_dir = save_part_segment + video_dir.split('/')[-2]
            create_dir(os.path.join(out_dir, save_segment_dir))
            for selected_txt in sorted(glob.glob(video_dir + file_exp)):
                vid_ind = selected_txt.split('/')[-1][:-4].split('_')[-1]
                imgs_dir = os.path.join(video_dir, 'CutVid_%s' % vid_ind)
                im_list = np.loadtxt(selected_txt, dtype=str, delimiter=',')
                current_result = _cityscape_folder(im_list, imgs_dir)
                out_file = os.path.join(
                    video_dir, 'cityscape_stats_s%s.txt', out_suffix
                )
                np.savetxt(
                    out_file, np.array(current_result), delimiter=',', fmt='%s'
                )
