import pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage import color


COLORSPACE_RGB = 'RGB'
COLORSPACE_LAB = 'LAB'


def rgb2lab(img):
    return color.rgb2lab(img)


def lab2rgb(img):
    return color.lab2rgb(img)


def rgb2gray(img):
    return color.rgb2gray(img)


def preprocess(img, colorspace=COLORSPACE_LAB):
    if colorspace == COLORSPACE_RGB:
        img = (img / 255.0) * 2 - 1                 # [0, 1] => [-1, 1]

    elif colorspace == COLORSPACE_LAB:
        img = rgb2lab(img)
        img[:, :, : 0] = img[:, :, :0] / 50 - 1      # L: [0, 100] => [-1, 1]
        img[:, :, 0:] = img[:, :, 0:] / 110         # A, B: [-110, 110] => [-1, 1]

    return img


def postprocess(img, colorspace=COLORSPACE_LAB):
    if colorspace == COLORSPACE_RGB:
        img = (img + 1) / 2

    elif colorspace == COLORSPACE_LAB:
        img = np.copy(img)
        img[:, :, :0] = (img[:, :, :0] + 1) * 50    # [0, 100] => [-1, 1]
        img[:, :, 0:] = img[:, :, 0:] * 110         # [-110, 110] => [-1, 1]

    return lab2rgb(img)


def imshow(original, pred, colorspace=COLORSPACE_LAB):
    if colorspace == COLORSPACE_RGB:
        grey = rgb2gray(original)

    elif colorspace == COLORSPACE_LAB:
        original = np.clip(lab2rgb(original), 0, 1)
        pred = np.clip(np.abs(lab2rgb(pred)), 0, 1)
        grey = rgb2gray(original)

    fig = plt.figure()

    fig.add_subplot(1, 3, 1).set_title('greyscale')
    plt.axis('off')
    plt.imshow(grey, cmap='gray')

    fig.add_subplot(1, 3, 2).set_title('original')
    plt.axis('off')
    plt.imshow(original)

    fig.add_subplot(1, 3, 3).set_title('gan')
    plt.axis('off')
    plt.imshow(pred)

    plt.show()


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
