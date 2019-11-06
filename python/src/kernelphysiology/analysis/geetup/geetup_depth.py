"""
Collection of functions to analyse the geetup related data.
"""

import os
import glob
import numpy as np

import cv2

from kernelphysiology.dl.geetup.geetup_utils import parse_gt_line
from kernelphysiology.utils.path_utils import create_dir


def _monodepth_folder(im_list, dir_path, bins=10):
    all_results = []
    # going through all the images of the list
    for frame_num in range(im_list.shape[0]):
        name = im_list[frame_num][0]

        img_depth = np.load(os.path.join(dir_path, name[:-3] + 'npy'))

        # Squeeze the dimensions (1, 1, ...)
        img_depth = np.squeeze(img_depth).copy()

        # Image needs to be resized (192, 640)
        img_depth = cv2.resize(
            img_depth, (640, 360), interpolation=cv2.INTER_NEAREST
        )

        # The gaze coordinates
        gaze = parse_gt_line(im_list[frame_num][1])

        pix_per_depth = []
        for key in range(bins):
            pix_per_depth.append(
                np.count_nonzero((img_depth >= key) & (img_depth < key + 1))
            )

        all_results.append(
            [im_list[frame_num][0], img_depth[gaze[0], gaze[1]], pix_per_depth]
        )
    return all_results


def report_monodepth(img_folder, txt_folder, out_dir=None, prefix_dir='npys',
                     out_name='depth_stats', in_exp='SELECTED_IMGS_*.txt'):
    if out_dir is None:
        out_dir = img_folder
    for part_dir in sorted(glob.glob(txt_folder + '/*/')):
        save_part = part_dir.split('/')[-2]
        create_dir(os.path.join(out_dir, save_part))
        save_part_segment = save_part + '/segments/'
        create_dir(os.path.join(out_dir, save_part_segment))
        for video_dir in sorted(glob.glob(part_dir + '/segments/*/')):
            print(video_dir)
            save_segment_dir = save_part_segment + video_dir.split('/')[-2]
            create_dir(os.path.join(out_dir, save_segment_dir))
            for selected_txt in sorted(glob.glob(video_dir + in_exp)):
                vid_ind = selected_txt.split('/')[-1][:-4].split('_')[-1]
                imgs_dir = os.path.join(
                    video_dir, 'CutVid_%s/%s' % (vid_ind, prefix_dir)
                )
                im_list = np.loadtxt(selected_txt, dtype=str, delimiter=',')
                # replacing the txt folder with img folder
                imgs_dir = imgs_dir.replace(txt_folder, img_folder)
                current_result = _monodepth_folder(im_list, imgs_dir)
                video_dir_save = video_dir.replace(txt_folder, img_folder)
                out_file = os.path.join(
                    video_dir_save, '%s_%s.txt' % (out_name, vid_ind)
                )
                header = 'im_name,gaze_depth,pixels_per_depth'
                np.savetxt(
                    out_file, np.array(current_result), delimiter=';', fmt='%s',
                    header=header
                )
