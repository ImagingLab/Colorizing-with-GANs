import pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage import color


COLORSPACE_RGB = 'RGB'
COLORSPACE_LAB = 'LAB'


def rgb2gray(img):
    return color.rgb2gray(img)


def preprocess(img, colorspace=COLORSPACE_LAB):
    if colorspace.upper() == COLORSPACE_RGB:
        img = (img / 255.0) * 2 - 1                 # [0, 1] => [-1, 1]

    elif colorspace.upper() == COLORSPACE_LAB:
        img = color.rgb2lab(img)
        img[..., 0] = img[..., 0] / 50 - 1          # L: [0, 100] => [-1, 1]
        img[..., 1:] = img[..., 1:] / 110           # A, B: [-110, 110] => [-1, 1]

    return img


def postprocess(img, colorspace=COLORSPACE_LAB):
    if colorspace.upper() == COLORSPACE_RGB:
        img = (img + 1) / 2

    elif colorspace.upper() == COLORSPACE_LAB:
        img = np.copy(img).astype(np.float64)
        img[..., 0] = (img[..., 0] + 1) * 50        # L: [-1, 1] => [0, 100]
        img[..., 1:] = img[..., 1:] * 110           # A, B: [-1, 1] => [-110, 110]

    if len(img.shape) == 4:
        img = np.array([color.lab2rgb(i) for i in img])
    else:
        img = color.lab2rgb(img)

    return img


def imshow(original, pred):
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
