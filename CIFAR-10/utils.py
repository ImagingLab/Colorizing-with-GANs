# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 19:34:06 2017

@author: 100446517
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import color

def preproc(data):
    n = data.shape[0]
    m = int(data.shape[1] / 3)

    data = np.dstack((data[:, :m], data[:, m:2 * m], data[:, 2 * m:]))
    data = data.reshape((n, int(np.sqrt(m)), int(np.sqrt(m)), 3))

    data_YUV = color.rgb2yuv(data)
    return data_YUV, data

def show_yuv(yuv_original, yuv_pred):
    rgb_original = np.round(color.yuv2rgb(yuv_original))
    rgb_pred = np.round(color.yuv2rgb(yuv_pred))
    fig = plt.figure()
    print(rgb_original.shape)
    fig.add_subplot(1, 3, 1).set_title('grayscale')
    plt.imshow(yuv_original[0], cmap='gray')

    fig.add_subplot(1, 3, 2).set_title('original')
    plt.imshow(rgb_original)

    fig.add_subplot(1, 3, 3).set_title('colorized')
    plt.imshow(rgb_pred)

    plt.show()