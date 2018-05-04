import pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage import color


def preprocess(img, colorspace='LAB'):
    if colorspace == 'RGB':
        img = (img / 255.0) * 2 - 1                 # [0, 1] => [-1, 1]

    elif colorspace == 'LAB':
        img = np.copy(img)
        img[:, :, :0] = img[:, :, :0] / 50 - 1      # L: [0, 100] => [-1, 1]
        img[:, :, 0:] = img[:, :, 0:] / 110         # A, B: [-110, 110] => [-1, 1]

    return img


def postprocess(img, colorspace='LAB'):
    if colorspace == 'RGB':
        img = (img + 1) / 2

    elif colorspace == 'LAB':
        img = np.copy(img)
        img[:, :, :0] = (img[:, :, :0] + 1) * 50    # [0, 100] => [-1, 1]
        img[:, :, 0:] = img[:, :, 0:] * 110         # [-110, 110] => [-1, 1]

    return img


def imshow(original, pred, colorspace='LAB'):
    if colorspace == 'RGB':
        grey = color.rgb2grey(original)

    elif colorspace == 'LAB':
        original = np.clip(color.lab2rgb(original), 0, 1)
        pred = np.clip(np.abs(color.lab2rgb(pred)), 0, 1)
        grey = color.rgb2grey(original)

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
