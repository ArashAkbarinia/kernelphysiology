import numpy as np
import math


def gaussian_width(sigma, max_width=100, threshold=1e-4):
    filter_width = 1
    for pw in range(np.floor(max_width / 2).astype('int'), -1, -1):
        if np.exp(-(pw ** 2) / (2 * sigma ** 2)) > threshold:
            filter_width = pw
            break
    return filter_width * 2 + 1


def gaussian_kernel2(sigmax, sigmay=None, meanx=0, meany=0, theta=0, width=None, threshold=1e-4):
    if sigmax == 0:
        return 1
    if sigmay is None:
        sigmay = sigmax
        max_sigma = sigmax
    else:
        max_sigma = np.maximum(sigmax, sigmay)
    if width is None:
        sizex = gaussian_width(sigma=max_sigma, max_width=100, threshold=threshold)
    else:
        sizex = width
    sizey = sizex
 
    centrex = (sizex + 1) / 2
    centrey = (sizey + 1) / 2
    centrex = centrex + (meanx * centrex)
    centrey = centrey + (meany * centrey)

    a =  math.cos(theta) ** 2 / 2 / sigmax ** 2 + math.sin(theta) ** 2 / 2 / sigmay ** 2
    b = -math.sin(2 * theta) / 4 / sigmax ** 2 + math.sin(2 * theta) / 4 / sigmay ** 2
    c =  math.sin(theta) ** 2 / 2 / sigmax ** 2 + math.cos(theta) ** 2 / 2 / sigmay ** 2

    kernel = np.zeros((sizex, sizey))
    for i in range(sizex):
        for j in range(sizey):
            x = (i + 1) * 1 - centrex
            y = 1 * (j + 1) - centrey
            kernel[i, j] = np.exp(-(a * x ** 2 + 2 * b * x * y + c * y ** 2))

    kernel /= kernel.sum()

    return kernel