import numpy as np

import cv2


def face_generator(
        template,
        eye_sep=10, eye_h=60, eye_w=12, eye_l=6, eye_t=2, eye_colour=(0, 0, 0),
        nose_h=100, nose_l=30, nose_t=2, nose_colour=(0, 0, 0),
        mouth_h=150, mouth_w=50, mouth_t=2, mouth_colour=(0, 0, 0),
):
    rows, cols = template.shape
    cen_row = int(rows / 2)
    cen_col = int(cols / 2)

    face_img = template.copy()
    face_img = np.repeat(face_img[:, :, np.newaxis], 3, axis=2)

    # nose
    nose_pt1 = (cen_col, nose_h)
    nose_pt2 = (nose_pt1[0], nose_pt1[1] + nose_l)
    face_img = cv2.line(face_img, nose_pt1, nose_pt2, nose_colour, thickness=nose_t)

    # mouth

    mouth_pt1 = (cen_col - int(mouth_w / 2), mouth_h)
    mouth_pt2 = (mouth_pt1[0] + mouth_w, mouth_h)
    face_img = cv2.line(face_img, mouth_pt1, mouth_pt2, mouth_colour, thickness=mouth_t)

    # eyes
    eye_axes_length = (eye_w, eye_l)
    l_eye_c = (cen_col - eye_w - eye_sep, eye_h)
    face_img = cv2.ellipse(face_img, l_eye_c, eye_axes_length, 0, 0, 360, eye_colour,
                           thickness=eye_t)

    r_eye_c = (cen_col + eye_w + eye_sep, eye_h)
    face_img = cv2.ellipse(face_img, r_eye_c, eye_axes_length, 0, 0, 360, eye_colour,
                           thickness=eye_t)

    return face_img
