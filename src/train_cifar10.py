import os
import time
import numpy as np
from keras.utils import generic_utils
from model import create_models
from dataset import load_cifar10_data, load_cfar10_test_data
from utils import show_lab


EPOCHS = 1000
BATCH_SIZE = 128
LEARNING_RATE = 0.002
MOMENTUM = 0.5
LAMBDA = 100
INPUT_SHAPE_GEN = (32, 32, 1)
INPUT_SHAPE_DIS = (32, 32, 4)
WEIGHTS_GEN = 'weights_cifar10_gen.hdf5'
WEIGHTS_DIS = 'weights_cifar10_dis.hdf5'
WEIGHTS_GAN = 'weights_cifar10_gan.hdf5'
MODE = 1  # 1: train - 2: visualize


model_gen, model_dis, model_gan = create_models(
    input_shape_gen=INPUT_SHAPE_GEN,
    input_shape_dis=INPUT_SHAPE_DIS,
    lr=LEARNING_RATE,
    momentum=MOMENTUM,
    l1_wight=LAMBDA)


if os.path.exists(WEIGHTS_GEN):
    model_gen.load_weights(WEIGHTS_GEN)

if os.path.exists(WEIGHTS_DIS):
    model_dis.load_weights(WEIGHTS_DIS)

if os.path.exists(WEIGHTS_GAN):
    model_gan.load_weights(WEIGHTS_GAN)


model_gen.summary()
model_dis.summary()
model_gan.summary()


if MODE == 1:
    data_lab, data_grey = load_cifar10_data(outType='LAB')

    print("Start training")
    for e in range(EPOCHS):
        batch_counter = 1
        batch_total = data_lab.shape[0] // BATCH_SIZE
        progbar = generic_utils.Progbar(batch_total * BATCH_SIZE)
        start = time.time()

        while batch_counter < batch_total:
            lab_batch = data_lab[(batch_counter - 1) * BATCH_SIZE:batch_counter * BATCH_SIZE]
            grey_batch = data_grey[(batch_counter - 1) * BATCH_SIZE:batch_counter * BATCH_SIZE]

            if batch_counter % 2 == 0:
                x_dis = np.concatenate((model_gen.predict(grey_batch), grey_batch), axis=3)
                y_dis = np.zeros((BATCH_SIZE, 1))
            else:
                x_dis = np.concatenate((lab_batch, grey_batch), axis=3)
                y_dis = np.ones((BATCH_SIZE, 1))

            dis_loss = model_dis.train_on_batch(x_dis, y_dis)

            model_dis.trainable = False
            x_gen = grey_batch
            y_gen = np.ones((BATCH_SIZE, 1))
            x_output = lab_batch
            gen_loss = model_gan.train_on_batch(x_gen, [y_gen, x_output])
            model_dis.trainable = True

            batch_counter += 1

            progbar.add(BATCH_SIZE,
                        values=[("D loss", dis_loss),
                                ("G loss", gen_loss[0]),
                                ("G L1", gen_loss[1])])

        print("")
        print('Epoch %s/%s, Time: %s' % (e + 1, EPOCHS, time.time() - start))
        model_gen.save_weights(WEIGHTS_GEN, overwrite=True)
        model_dis.save_weights(WEIGHTS_DIS, overwrite=True)
        model_gan.save_weights(WEIGHTS_GAN, overwrite=True)

elif MODE == 2:
    data_test_lab, data_test_grey = load_cfar10_test_data(outType='LAB')
    for i in range(0, 5000):
        grey = data_test_grey[i]
        lab_original = data_test_lab[i]
        lab_pred = np.array(model_gen.predict(grey[None, :, :, :]))[0]
        show_lab(lab_original, lab_pred)
