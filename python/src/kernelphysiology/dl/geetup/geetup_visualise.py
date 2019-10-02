"""
Collection of visualisation function related to GEETUP.
"""

import numpy as np

import cv2

from kernelphysiology.utils.imutils import max_pixel_ind


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
