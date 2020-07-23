"""
Collection of visualisation function related to GEETUP.
"""

import numpy as np

import cv2

from kernelphysiology.dl.geetup import geetup_utils
from kernelphysiology.dl.geetup.geetup_utils import map_point_to_image_size
from kernelphysiology.dl.pytorch.geetup import geetup_db
from kernelphysiology.utils import path_utils
from kernelphysiology.utils.imutils import max_pixel_ind


def draw_circle_clips(preds, geetup_info, beg_ind, end_ind, out_dir):
    path_utils.create_dir(out_dir)
    for j in range(beg_ind, end_ind):
        f_path, f_gt = geetup_info.__getitem__(j)
        f_path = f_path[-1]
        f_gt = f_gt[0]
        f_pred = preds[j]
        # read the image
        img_in = cv2.imread(f_path)
        # draw the gt
        img_in = cv2.circle(
            img_in, (f_gt[1], f_gt[0]), 5, (0, 255, 0), thickness=9
        )
        # draw the prediction
        pred = np.int(f_pred)
        # TODO: support other image sizes
        pred = map_point_to_image_size(pred, (360, 640), (180, 320))
        pred[0] = int(pred[0])
        pred[1] = int(pred[1])
        img_in = cv2.circle(
            img_in, (pred[1], pred[0]), 5, (0, 0, 255), thickness=9
        )
        img_name = f_path.split('/')[-1]
        out_file = '%s/%s' % (out_dir, img_name)
        cv2.imwrite(out_file, img_in)


def clip_visualise(db_path, pred_path, euc_path, out_dir,
                   video_clips_inds=None):
    preds = path_utils.read_pickle(pred_path)
    eucs = path_utils.read_pickle(euc_path)
    geetup_info = geetup_db.GeetupDatasetInformative(db_path)

    if video_clips_inds is None:
        video_clips_inds = geetup_utils.get_video_clips_inds(geetup_info)

    for clip_inds in video_clips_inds:
        beg_ind, end_ind = clip_inds
        print(
            'Video [%d %d] %.2f' % (
                beg_ind, end_ind, np.median(eucs[beg_ind:end_ind])
            )
        )
        draw_circle_clips(preds, geetup_info, beg_ind, end_ind, out_dir)


def draw_circle_results(img_org, gt, pred, out_file=None):
    # OpenCV goes BGR
    img_out = img_org.copy()[:, :, ::-1]
    rows, cols, _ = img_out.shape
    gt_pixel = max_pixel_ind(
        np.reshape(gt.squeeze(), (rows, cols))
    )
    # [::-1] because OpenCV point is XY, which is opposite of rows, cols
    img_out = cv2.circle(img_out, gt_pixel[::-1], 15, (0, 255, 0))

    pred_pixel = max_pixel_ind(
        np.reshape(pred.squeeze(), (rows, cols))
    )
    img_out = cv2.circle(img_out, pred_pixel[::-1], 15, (0, 0, 255))

    euc_dis = np.linalg.norm(np.asarray(pred_pixel) - np.asarray(gt_pixel))
    cx = round(cols / 2)
    cy = round(rows / 2)
    img_out = cv2.putText(
        img_out, str(int(euc_dis)), (cx, cy),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255)
    )
    if out_file is not None:
        cv2.imwrite(out_file, img_out)
    return img_out


def save_heatmap(heatmap, out_file):
    heatmap = heatmap.copy()
    heatmap /= heatmap.max()
    heatmap *= 255
    cv2.imwrite(out_file, np.uint8(heatmap))
