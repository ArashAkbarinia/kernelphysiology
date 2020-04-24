import numpy as np


def rigid_transform(pts1, pts2):
    assert len(pts1) == len(pts2)

    rows1, cols1 = pts1.shape
    rows2, cols2 = pts2.shape

    # find mean column wise
    cen1 = np.mean(pts1, axis=1)
    cen2 = np.mean(pts2, axis=1)

    if cols1 > cols2:
        new_b = np.ones((rows2, cols1))
        new_b[:, :cols2] = pts2
        d_cols = cols1 - cols2
        new_b[:, cols2:] = np.broadcast_to(cen2, (d_cols, rows2)).transpose()
        pts2 = new_b
        rows2, cols2 = pts2.shape
        cen2 = np.mean(pts2, axis=1)
    elif cols1 < cols2:
        new_a = np.ones((rows1, cols2))
        new_a[:, :cols1] = pts1
        d_cols = cols2 - cols1
        new_a[:, cols1:] = np.broadcast_to(cen1, (d_cols, rows1)).transpose()
        pts1 = new_a
        rows1, cols1 = pts1.shape
        cen1 = np.mean(pts1, axis=1)

    # subtract mean
    pts1_centred = pts1 - np.broadcast_to(cen1, (cols1, rows1)).transpose()
    pts2_centred = pts2 - np.broadcast_to(cen2, (cols2, rows2)).transpose()

    hmat = np.matmul(pts1_centred, pts2_centred.transpose())

    # sanity check
    # print(np.linalg.matrix_rank(hmat))
    # if np.linalg.matrix_rank(hmat) < rows_a:
    #     raise ValueError(
    #         "rank of H = {}, expecting {}".format(
    #             np.linalg.matrix_rank(hmat), rows_a
    #         )
    #     )

    # find rotation
    u, s, vt = np.linalg.svd(hmat)
    vut = np.matmul(vt.T, u.T)
    ones = np.eye(vt.T.shape[1])
    ones[-1, -1] = np.linalg.det(vut)
    rot_mat = np.matmul(np.matmul(vt.T, ones), u.T)

    # special reflection case
    # if np.linalg.det(rot_mat) < 0:
    #     print("det(R) < R, reflection detected!, correcting for it ...\n")
    #     vt[-1, :] *= -1
    #     rot_mat = np.matmul(vt.T, u.T)

    trans_vec = np.matmul(-rot_mat, cen1) + cen2

    return rot_mat, trans_vec


def apply_rigid_transform(rot_mat, trans_vec, pts):
    if rot_mat is not None:
        rotated_pts = np.matmul(rot_mat, pts)
    else:
        rotated_pts = pts
    if trans_vec is not None:
        trans_mat = np.broadcast_to(
            trans_vec, (pts.shape[1], trans_vec.shape[0])
        ).transpose()
        out = rotated_pts + trans_mat
    else:
        out = rotated_pts
    return out


def affine_transform(pts1, pts2):
    # Compute the affine transformation using homogenous coordinates
    hom_pts1 = np.vstack([pts1, np.ones(len(pts1.T))])
    hom_pts2 = np.vstack([pts2, np.ones(len(pts2.T))])

    affine_mat = np.linalg.lstsq(hom_pts1.T, hom_pts2.T, rcond=None)[0]

    return affine_mat.T


def apply_affine_transform(affine_mat, pts):
    hom_pts = np.vstack([pts, np.ones(len(pts.T))])
    tmp = np.matmul(affine_mat, hom_pts)
    out_pts = np.array([x[:-1] / x[-1] for x in tmp.T])

    return out_pts.T
