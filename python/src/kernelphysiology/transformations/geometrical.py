import numpy as np


def rigid_transform(points_a, points_b):
    assert len(points_a) == len(points_b)

    rows_a, cols_a = points_a.shape
    rows_b, cols_b = points_b.shape

    # find mean column wise
    cen_a = np.mean(points_a, axis=1)
    cen_b = np.mean(points_b, axis=1)

    if cols_a > cols_b:
        new_b = np.ones((rows_b, cols_a))
        new_b[:, :cols_b] = points_b
        d_cols = cols_a - cols_b
        new_b[:, cols_b:] = np.broadcast_to(cen_b, (d_cols, rows_b)).transpose()
        points_b = new_b
        rows_b, cols_b = points_b.shape
        cen_b = np.mean(points_b, axis=1)
    elif cols_a < cols_b:
        new_a = np.ones((rows_a, cols_b))
        new_a[:, :cols_a] = points_a
        d_cols = cols_b - cols_a
        new_a[:, cols_a:] = np.broadcast_to(cen_a, (d_cols, rows_a)).transpose()
        points_a = new_a
        rows_a, cols_a = points_a.shape
        cen_a = np.mean(points_a, axis=1)

    # subtract mean
    acentred = points_a - np.broadcast_to(cen_a, (cols_a, rows_a)).transpose()
    bcentred = points_b - np.broadcast_to(cen_b, (cols_b, rows_b)).transpose()

    hmat = np.matmul(acentred, bcentred.transpose())

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

    trans_vec = np.matmul(-rot_mat, cen_a) + cen_b

    return rot_mat, trans_vec
