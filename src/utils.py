# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from skimage import color


def preproc(data, normalize=False, flip=False, mean_image=None, outType='YUV'):
    data_size = data.shape[0]
    img_size = int(data.shape[1] / 3)

    if normalize:
        if mean_image is None:
            mean_image = np.mean(data)

        mean_image = mean_image / np.float32(255)
        data = (data - mean_image) / np.float32(255)

    data_RGB = np.dstack((data[:, :img_size], data[:, img_size:2 * img_size], data[:, 2 * img_size:]))
    data_RGB = data_RGB.reshape((data_size, int(np.sqrt(img_size)), int(np.sqrt(img_size)), 3))

    if flip:
        data_RGB = data_RGB[0:data_size, :, :, :]
        data_RGB_flip = data_RGB[:, :, :, ::-1]
        data_RGB = np.concatenate((data_RGB, data_RGB_flip), axis=0)

    if outType == 'YUV':
        data_out = color.rgb2yuv(data_RGB)
        return data_out, data_RGB  # returns YUV as 4D tensor and RGB as 4D tensor

    elif outType == 'LAB':
        data_out = color.rgb2lab(data_RGB)
        data_gray = color.rgb2gray(data_RGB)[:, :, :, None]
        return data_out, data_gray  # returns LAB and grayscale as 4D tensor



def show_yuv(yuv_original, yuv_pred):
    rgb_original = np.clip(color.yuv2rgb(yuv_original), 0, 1)
    rgb_pred = np.clip(np.abs(color.yuv2rgb(yuv_pred)), 0, 1)
    grey = color.rgb2grey(yuv_original)

    fig = plt.figure()
    fig.add_subplot(1, 3, 1).set_title('greyscale')
    plt.axis('off')
    plt.imshow(grey, cmap='gray')

    fig.add_subplot(1, 3, 2).set_title('original')
    plt.axis('off')
    plt.imshow(rgb_original)

    fig.add_subplot(1, 3, 3).set_title('gan')
    plt.axis('off')
    plt.imshow(rgb_pred)

    plt.show()


def show_rgb(rgb_original, rgb_pred):
    grey = color.rgb2grey(rgb_original)

    fig = plt.figure()
    fig.add_subplot(1, 3, 1).set_title('greyscale')
    plt.axis('off')
    plt.imshow(grey, cmap='gray')

    fig.add_subplot(1, 3, 2).set_title('original')
    plt.axis('off')
    plt.imshow(rgb_original)

    fig.add_subplot(1, 3, 3).set_title('gan')
    plt.axis('off')
    plt.imshow(rgb_pred)

    plt.show()


def show_lab(lab_original, lab_pred):
    lab_pred = lab_pred.astype(np.float64)
    rgb_original = np.clip(color.lab2rgb(lab_original), 0, 1)
    rgb_pred = np.clip(np.abs(color.lab2rgb(lab_pred)), 0, 1)
    grey = color.rgb2grey(lab_original)

    fig = plt.figure()
    fig.add_subplot(1, 3, 1).set_title('greyscale')
    plt.axis('off')
    plt.imshow(grey, cmap='gray')

    fig.add_subplot(1, 3, 2).set_title('original')
    plt.axis('off')
    plt.imshow(rgb_original)

    fig.add_subplot(1, 3, 3).set_title('gan')
    plt.axis('off')
    plt.imshow(rgb_pred)

    plt.show()
