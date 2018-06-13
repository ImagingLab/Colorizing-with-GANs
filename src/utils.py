import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def stitch_images(grayscale, original, pred):
    gap = 5
    width, height = original[0][:, :, 0].shape
    img_per_row = 2 if width > 200 else 4
    img = Image.new('RGB', (width * img_per_row * 3 + gap * (img_per_row - 1), height * int(len(original) / img_per_row)))

    grayscale = np.array(grayscale).squeeze()
    original = np.array(original)
    pred = np.array(pred)

    for ix in range(len(original)):
        xoffset = int(ix % img_per_row) * width * 3 + int(ix % img_per_row) * gap
        yoffset = int(ix / img_per_row) * height
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


def moving_average(data, window_width):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    return ma_vec


def imshow(img, title=''):
    fig = plt.gcf()
    fig.canvas.set_window_title(title)
    plt.axis('off')
    plt.imshow(img, interpolation='none')
    plt.show()


def turing_test(real_img, fake_img, delay=0):
    height, width, _ = real_img.shape
    imgs = np.array([real_img, (fake_img * 255).astype(np.uint8)])
    real_index = np.random.binomial(1, 0.5)
    fake_index = (real_index + 1) % 2

    img = Image.new('RGB', (2 + width * 2, height))
    img.paste(Image.fromarray(imgs[real_index]), (0, 0))
    img.paste(Image.fromarray(imgs[fake_index]), (2 + width, 0))

    img.success = 0

    def onclick(event):
        if event.xdata is not None:
            if event.x < width and real_index == 0:
                img.success = 1

            elif event.x > width and real_index == 1:
                img.success = 1

        plt.gcf().canvas.stop_event_loop()

    plt.ion()
    plt.gcf().canvas.mpl_connect('button_press_event', onclick)
    plt.title('click on the real image')
    plt.axis('off')
    plt.imshow(img, interpolation='none')
    plt.show()
    plt.draw()
    plt.gcf().canvas.start_event_loop(delay)

    return img.success


def visualize(train_log_file, test_log_file, window_width, title=''):
    train_data = np.loadtxt(train_log_file)
    test_data = np.loadtxt(test_log_file)

    if len(train_data.shape) < 2:
        return

    if len(train_data) < window_width:
        window_width = len(train_data) - 1

    fig = plt.gcf()
    fig.canvas.set_window_title(title)

    plt.ion()
    plt.subplot('121')
    plt.cla()
    if len(train_data) > 1:
        plt.plot(moving_average(train_data[:, 8], window_width))
    plt.title('train')

    plt.subplot('122')
    plt.cla()
    if len(test_data) > 1:
        plt.plot(test_data[:, 8])
    plt.title('test')

    plt.show()
    plt.draw()
    plt.pause(.01)
