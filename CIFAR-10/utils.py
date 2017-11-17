# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 19:34:06 2017

@author: 100446517
"""

import numpy as np
import matplotlib.pyplot as plt


def rgb2yuv(data):
    M = np.array(([0.2126, 0.7152, 0.0722], [-0.09991, -0.33609, 0.436], [0.615, -0.55861, -0.05639]))

    n = data.shape[0]
    m = int(data.shape[1] / 3)

    R = data[:, :m].reshape([1, -1])
    G = data[:, m:2 * m].reshape([1, -1])
    B = data[:, 2 * m:].reshape([1, -1])

    data_YUV = np.dot(M, np.vstack((R, G, B)))

    Y = data_YUV[0, :].reshape([m, n])
    U = data_YUV[1, :].reshape([m, n])
    V = data_YUV[2, :].reshape([m, n])

    return np.hstack((Y,U,V))


def yuv2rgb(data):
    M = np.array(([1, 0, 1.28033], [1, -0.21482, -0.38059], [1, 2.12798, 0]))

    n = data.shape[0]
    m = int(data.shape[1] / 3)

    Y = data[:, :m].reshape([1, -1])
    U = data[:, m:2 * m].reshape([1, -1])
    V = data[:, 2 * m:].reshape([1, -1])

    data_RGB = np.dot(M, np.vstack((Y, U, V)))

    R = data_RGB[0, :].reshape([m, n])
    G = data_RGB[1, :].reshape([m, n])
    B = data_RGB[2, :].reshape([m, n])

    return np.hstack((R,G,B))

def data_vis(data):
    n = data.shape[0]
    m = int(data.shape[1] / 3)

    Y = data[:, :m].reshape([1, -1])
    U = data[:, m:2 * m].reshape([1, -1])
    V = data[:, 2 * m:].reshape([1, -1])

    out_data = np.dstack((data[:,:m], data[:,m:2*m], data[:,2*m:]))
    out_data = out_data.reshape((n, int(np.sqrt(m)), int(np.sqrt(m)), 3))

    return out_data

def show_yuv(yuv_original, yuv_pred):
    rgb_original = np.round(yuv2rgb(yuv_original))
    rgb_pred = np.round(yuv2rgb(yuv_pred))
    fig = plt.figure()
    print(rgb_original.shape)
    fig.add_subplot(1, 3, 1).set_title('grayscale')
    plt.imshow(yuv_original[0], cmap='gray')

    fig.add_subplot(1, 3, 2).set_title('original')
    plt.imshow(rgb_original)

    fig.add_subplot(1, 3, 3).set_title('colorized')
    plt.imshow(rgb_pred)

    plt.show()