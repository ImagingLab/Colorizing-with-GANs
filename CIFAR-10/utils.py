# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 19:34:06 2017

@author: 100446517
"""

import numpy as np


def rgb2yuv(data):
    # m x (3 x 64 x 64)
    n = data.shape[0]
    m = int(data.shape[1] / 3)
    # Conversion matrix: rgb2yuv under BT.709 standard
    M = np.array(([0.2126, 0.7152, 0.0722], [-0.09991, -0.33609, 0.436], [0.615, -0.55861, -0.05639]))

    R = data[:, :m].reshape([1, -1])
    G = data[:, m:2 * m].reshape([1, -1])
    B = data[:, 2 * m:].reshape([1, -1])

    # data_RGB = np.vstack((R,G,B))
    data_YUV = np.dot(M, np.vstack((R, G, B)))

    # Y = data_YUV[0,:].reshape([64, 64, n])
    # U = data_YUV[1,:].reshape([64, 64, n])
    # V = data_YUV[2,:].reshape([64, 64, n])
    Y = data_YUV[0, :].reshape([m, n])
    U = data_YUV[1, :].reshape([m, n])
    V = data_YUV[2, :].reshape([m, n])

    # YUV = np.zeros([3, 64, 64, n])
    # YUV[0,:,:,:] = Y
    # YUV[1,:,:,:] = U
    # YUV[2,:,:,:] = V
    YUV = np.zeros([3, m, n])
    YUV[0, :, :] = Y
    YUV[1, :, :] = U
    YUV[2, :, :] = V

    # return YUV
    # return np.transpose(YUV, (3,0,1,2))
    return np.transpose(YUV, (2, 0, 1))


# yuv2rgb only works if rgb2rgb returns YUV instead of the transposed version
def yuv2rgb(data):
    M = np.array(([1, 0, 1.28033], [1, -0.21482, -0.38059], [1, 2.12798, 0]))
    m = data.shape[2]

    data = np.transpose(data, (1, 2, 0))
    Y = data[0, :, :].reshape([1, -1])
    U = data[1, :, :].reshape([1, -1])
    V = data[2, :, :].reshape([1, -1])

    data_RGB = np.dot(M, np.vstack((Y, U, V)))

    R = data_RGB[0, :].reshape([-1, m])
    G = data_RGB[1, :].reshape([-1, m])
    B = data_RGB[2, :].reshape([-1, m])

    RGB = np.hstack((R, G, B))
    return RGB
