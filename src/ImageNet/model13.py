import os
import numpy as np
import keras.backend as K
from keras import metrics
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Input, MaxPool2D, Activation, BatchNormalization, UpSampling2D, concatenate, LeakyReLU, Conv2D
from scipy.misc import imread
from skimage import color
import os, sys, inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))

from dataset import *
from utils import *

IMAGENET_BATCH_SIZE = 128116
EPOCHS = 100
BATCH_SIZE = 64
VALIDATION_SIZE = 4096
LEARNING_RATE = 0.001
INPUT_SHAPE = (64, 64, 1)
WEIGHTS = 'model13.hdf5'
MODE = 1  # 1: train - 2: visualize


def eacc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))


def mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


def mae(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)


def learning_scheduler(epoch):
    lr = LEARNING_RATE / (2 ** (epoch // 20))
    print('\nlearning rate: ' + str(lr) + '\n')
    return lr


def create_conv(filters, kernel_size, inputs, name=None, strides=(1, 1), bn=True, padding='same', activation='relu'):
    conv = Conv2D(filters, kernel_size, strides=strides, padding=padding, kernel_initializer='he_normal', name=name)(
        inputs)

    if bn == True:
        conv = BatchNormalization()(conv)

    if activation == 'relu':
        conv = Activation(activation)(conv)
    elif activation == 'leakyrelu':
        conv = LeakyReLU()(conv)

    return conv


def create_model():
    inputs = Input(INPUT_SHAPE)
    conv1 = create_conv(64, (3, 3), inputs, 'conv1_1', activation='leakyrelu')
    conv1 = create_conv(64, (3, 3), conv1, 'conv1_2', activation='leakyrelu')
    # pool1 = create_conv(64, (2, 2), conv1, 'conv1_3', activation='leakyrelu', padding='valid', strides=(2, 2))
    pool1 = MaxPool2D((2, 2))(conv1)

    conv2 = create_conv(128, (3, 3), pool1, 'conv2_1', activation='leakyrelu')
    conv2 = create_conv(128, (3, 3), conv2, 'conv2_2', activation='leakyrelu')
    # pool2 = create_conv(128, (2, 2), conv2, 'conv2_3', activation='leakyrelu', padding='valid', strides=(2, 2))
    pool2 = MaxPool2D((2, 2))(conv2)

    conv3 = create_conv(256, (3, 3), pool2, 'conv3_1', activation='leakyrelu')
    conv3 = create_conv(256, (3, 3), conv3, 'conv3_2', activation='leakyrelu')
    # pool3 = create_conv(256, (2, 2), conv3, 'conv3_3', activation='leakyrelu', padding='valid', strides=(2, 2))
    pool3 = MaxPool2D((2, 2))(conv3)

    conv4 = create_conv(512, (3, 3), pool3, 'conv4_1', activation='leakyrelu')
    conv4 = create_conv(512, (3, 3), conv4, 'conv4_2', activation='leakyrelu')
    # pool4 = create_conv(512, (2, 2), conv4, 'conv4_3', activation='leakyrelu', padding='valid', strides=(2, 2))
    pool4 = MaxPool2D((2, 2))(conv4)

    conv5 = create_conv(1024, (3, 3), pool4, 'conv5_1', activation='leakyrelu')
    conv5 = create_conv(1024, (3, 3), conv5, 'conv5_2', activation='leakyrelu')

    up6 = create_conv(512, (2, 2), UpSampling2D((2, 2))(conv5), 'up6', activation='relu')
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = create_conv(512, (3, 3), merge6, 'conv6_1', activation='relu')
    conv6 = create_conv(512, (3, 3), conv6, 'conv6_2', activation='relu')

    up7 = create_conv(256, (2, 2), UpSampling2D((2, 2))(conv6), 'up7', activation='relu')
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = create_conv(256, (3, 3), merge7, 'conv7_1', activation='relu')
    conv7 = create_conv(256, (3, 3), conv7, 'conv7_2', activation='relu')

    up8 = create_conv(128, (2, 2), UpSampling2D((2, 2))(conv7), 'up8', activation='relu')
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = create_conv(128, (3, 3), merge8, 'conv8_1', activation='relu')
    conv8 = create_conv(128, (3, 3), conv8, 'conv8_2', activation='relu')

    up9 = create_conv(64, (2, 2), UpSampling2D((2, 2))(conv8), 'up9', activation='relu')
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = create_conv(64, (3, 3), merge9, 'conv9_1', activation='relu')
    conv9 = create_conv(64, (3, 3), conv9, 'conv9_2', activation='relu')
    conv9 = Conv2D(3, (1, 1), padding='same', name='conv9_3')(conv9)

    model = Model(inputs=inputs, outputs=conv9)
    model.compile(optimizer=Adam(lr=LEARNING_RATE, beta_1=0.9),
                  loss='mean_squared_error',
                  metrics=['accuracy', eacc, mse, mae])

    if os.path.exists(WEIGHTS):
        model.load_weights(WEIGHTS)

    return model


if MODE == 1:
    model = create_model()
    model.summary()
    model_checkpoint = ModelCheckpoint(
        filepath=WEIGHTS,
        monitor='loss',
        verbose=1,
        save_best_only=True)

    scheduler = LearningRateScheduler(learning_scheduler)

    model.fit_generator(
        imagenet_data_generator(batch_size=BATCH_SIZE, flip=False, outType='LAB'),
        steps_per_epoch=10 * (IMAGENET_BATCH_SIZE // BATCH_SIZE),
        validation_data=imagenet_test_data_generator(batch_size=BATCH_SIZE, count=VALIDATION_SIZE, outType='LAB'),
        validation_steps=VALIDATION_SIZE // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[model_checkpoint, scheduler])

elif MODE == 2:
    model = create_model()
    data_test_lab, data_test_grey = load_imagenet_test_data(count=VALIDATION_SIZE, outType='LAB')

    for i in range(0, VALIDATION_SIZE):
        lab_original = data_test_lab[i]
        lab_pred = np.array(model.predict(data_test_grey[i:i+1]))[0]
        show_lab(lab_original, lab_pred)

elif MODE == 3:
    img = imread('290982-64.png')[:,:,0:3]
    INPUT_SHAPE = (img.shape[0], img.shape[1], 1)
    model = create_model()

    lab_pred = np.array(model.predict(color.rgb2grey(img)[None,:,:,None]))[0]
    show_lab(color.rgb2lab(img), lab_pred)
