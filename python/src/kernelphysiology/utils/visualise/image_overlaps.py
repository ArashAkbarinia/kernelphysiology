"""

"""

import cv2
from matplotlib import pyplot as plt


def intensity_overlap(img, target_size=None):
    overlap_img = img.copy()
    if len(overlap_img.shape) > 2:
        overlap_img = cv2.cvtColor(overlap_img, cv2.COLOR_RGB2GRAY)

    if target_size is None:
        target_size = (48, 48)
    overlap_img = cv2.resize(overlap_img, target_size, interpolation=cv2.INTER_CUBIC)

    fig_size = (32, 32)
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(overlap_img, cmap='gray')

    for i in range((overlap_img.shape[0])):
        for j in range((overlap_img.shape[1])):
            intensity = overlap_img[i, j]
            color = [1 - float(intensity) / 255] * 3
            text = ax.text(
                j, i, intensity, ha="center", va="center",
                color=color, fontsize=14
            )
    plt.axis('off')

    return fig
