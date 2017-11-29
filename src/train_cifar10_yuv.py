import os
import time
import numpy as np
from keras.utils import generic_utils
from model import create_models
from dataset import load_cifar10_data, load_cifar10_test_data, load_extra_data
from utils import show_yuv

EPOCHS = 500
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
MOMENTUM = 0.5
LAMBDA1 = 1
LAMBDA2 = 10
INPUT_SHAPE_GEN = (32, 32, 1)
INPUT_SHAPE_DIS = (32, 32, 3)
WEIGHTS_GEN = 'weights_cifar10_yuv_gen.hdf5'
WEIGHTS_DIS = 'weights_cifar10_yuv_dis.hdf5'
WEIGHTS_GAN = 'weights_cifar10_yuv_gan.hdf5'
MODE = 1  # 1: train - 2: visualize

model_gen, model_dis, model_gan = create_models(
    input_shape_gen=INPUT_SHAPE_GEN,
    input_shape_dis=INPUT_SHAPE_DIS,
    output_channels=2,
    lr=LEARNING_RATE,
    momentum=MOMENTUM,
    loss_weights=[LAMBDA1, LAMBDA2])

if os.path.exists(WEIGHTS_GEN):
    model_gen.load_weights(WEIGHTS_GEN)

if os.path.exists(WEIGHTS_DIS):
    model_dis.load_weights(WEIGHTS_DIS)

if os.path.exists(WEIGHTS_GAN):
    model_gan.load_weights(WEIGHTS_GAN)

model_gen.summary()
model_dis.summary()
model_gan.summary()


data_yuv, data_rgb = load_cifar10_data(outType='YUV')
data_test_yuv, data_test_rgb = load_cifar10_test_data(outType='YUV')
#data_extra_yuv, data_extra_rgb = load_extra_data(outType='YUV')
#data_yuv = np.concatenate((data_extra_yuv, data_yuv), axis=0)
#data_yuv = np.concatenate((data_yuv, data_test_yuv), axis=0)

data_yuv = data_yuv * 255
data_test_yuv = data_test_yuv * 255

data_y = data_yuv[:, :, :, :1]
data_uv = data_yuv[:, :, :, 1:]

data_test_y = data_test_yuv[:, :, :, :1]
data_test_uv = data_test_yuv[:, :, :, 1:]


if MODE == 1:
    print("Start training")
    for e in range(EPOCHS):
        batch_counter = 1
        toggle = True
        batch_total = data_yuv.shape[0] // BATCH_SIZE
        progbar = generic_utils.Progbar(batch_total * BATCH_SIZE)
        start = time.time()
        dis_res = 0

        while batch_counter < batch_total:
            uv_batch = data_uv[(batch_counter - 1) * BATCH_SIZE:batch_counter * BATCH_SIZE]
            y_batch = data_y[(batch_counter - 1) * BATCH_SIZE:batch_counter * BATCH_SIZE]

            batch_counter += 1

            toggle = not toggle
            if toggle:
                x_dis = np.concatenate((model_gen.predict(y_batch), y_batch), axis=3)
                y_dis = np.zeros((BATCH_SIZE, 1))
            else:
                x_dis = np.concatenate((uv_batch, y_batch), axis=3)
                y_dis = np.ones((BATCH_SIZE, 1))
                y_dis = np.random.uniform(low=0.9, high=1, size=BATCH_SIZE)

            dis_res = model_dis.train_on_batch(x_dis, y_dis)

            model_dis.trainable128 = False
            x_gen = y_batch
            y_gen = np.ones((BATCH_SIZE, 1))
            x_output = uv_batch
            gan_res = model_gan.train_on_batch(x_gen, [y_gen, x_output])
            model_dis.trainable = True

            progbar.add(BATCH_SIZE,
                        values=[("D loss", dis_res),
                                ("G total loss", gan_res[0]),
                                ("G loss", gan_res[1]),
                                ("G L1", gan_res[2]),
                                ("pacc", gan_res[5]),
                                ("acc", gan_res[6])])

        print("")
        print('Epoch %s/%s, Time: %s' % (e + 1, EPOCHS, round(time.time() - start)))
        if e % 10 == 0:
            ev = model_gan.evaluate(data_test_y, [np.ones((data_test_y.shape[0], 1)), data_test_uv])
            ev = np.round(np.array(ev), 4)
            print('G total loss: %s - G loss: %s - G L1: %s: pacc: %s - acc: %s' % (ev[0], ev[1], ev[2], ev[5], ev[6]))
        print('')
        model_gen.save_weights(WEIGHTS_GEN, overwrite=True)
        model_dis.save_weights(WEIGHTS_DIS, overwrite=True)
        model_gan.save_weights(WEIGHTS_GAN, overwrite=True)


elif MODE == 2:
    for i in range(0, 5000):
        print(i)
        y = data_test_y[i]
        yuv_original = data_test_yuv[i]
        uv_pred = np.array(model_gen.predict(y[None, :, :, :]))[0]
        yuv_pred = np.r_[(y.T, uv_pred.T[:1], uv_pred.T[1:])].T
        show_yuv(yuv_original / 255, yuv_pred / 255)