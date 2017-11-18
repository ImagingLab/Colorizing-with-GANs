# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 19:34:06 2017

@author: 100446517
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import color


def preproc(data, normalize=False):
    n = data.shape[0]
    m = int(data.shape[1] / 3)

    if normalize:
        image_mean = np.mean(data)
        mean_mat = image_mean * np.ones(data.shape)
        data = (data - mean_mat) / 255

    data_RGB = np.dstack((data[:, :m], data[:, m:2 * m], data[:, 2 * m:]))
    data_RGB = data_RGB.reshape((n, int(np.sqrt(m)), int(np.sqrt(m)), 3))

    data_YUV = color.rgb2yuv(data_RGB)
    data_grey = color.rgb2grey(data_RGB)

    return data_YUV, data_RGB, data_grey  # returns YUV as 4D tensor and RGB as 4D tensor


def show_yuv(yuv_original, yuv_pred):
    rgb_original = color.yuv2rgb(yuv_original)
    rgb_pred = np.abs(color.yuv2rgb(yuv_pred))
    grey = color.rgb2grey(yuv_original)

    fig = plt.figure()
    fig.add_subplot(1, 3, 1).set_title('greyscale')
    plt.imshow(grey, cmap='gray')

    fig.add_subplot(1, 3, 2).set_title('original')
    plt.imshow(rgb_original)

    fig.add_subplot(1, 3, 3).set_title('colorized')
    plt.imshow(rgb_pred)

    plt.show()


def show_rgb(rgb_original, rgb_pred):
    grey = color.rgb2grey(rgb_original)

    fig = plt.figure()
    fig.add_subplot(1, 3, 1).set_title('greyscale')
    plt.imshow(grey, cmap='gray')

    fig.add_subplot(1, 3, 2).set_title('original')
    plt.imshow(rgb_original)

    fig.add_subplot(1, 3, 3).set_title('colorized')
    plt.imshow(rgb_pred)

    plt.show()