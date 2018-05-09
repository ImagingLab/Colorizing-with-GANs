import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def stitch_images(grayscale, original, pred):
    width, height = original[0][:, :, 0].shape
    ROW = 2 if width > 200 else 4
    img = Image.new('RGB', (width * ROW * 3 + 10 * (ROW - 1), height * int(len(original) / ROW)))

    grayscale = np.array(grayscale).squeeze()
    original = np.array(original)
    pred = np.array(pred)

    for ix in range(len(original)):
        xoffset = int(ix % ROW) * width * 3 + int(ix % ROW) * 10
        yoffset = int(ix / ROW) * height
        im1 = Image.fromarray(grayscale[ix])
        im2 = Image.fromarray(original[ix])
        im3 = Image.fromarray((pred[ix] * 255).astype(np.uint8))
        img.paste(im1, (xoffset, yoffset))
        img.paste(im2, (xoffset + width, yoffset))
        img.paste(im3, (xoffset + width + width, yoffset))

    return img


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def imshow(original, pred):
    fig = plt.figure()

    fig.add_subplot(1, 2, 1).set_title('original')
    plt.axis('off')
    plt.imshow(original)

    fig.add_subplot(1, 2, 2).set_title('gan')
    plt.axis('off')
    plt.imshow(pred)

    plt.show()
