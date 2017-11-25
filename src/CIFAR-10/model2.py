import os
import numpy as np
import keras.backend as K
from keras import metrics
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Input, MaxPool2D, Activation, BatchNormalization, UpSampling2D, concatenate, LeakyReLU, Conv2D
import os, sys, inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))

from dataset import *
from utils import *

EPOCHS = 500
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
INPUT_SHAPE = (32, 32, 1)
WEIGHTS = 'model2.hdf5'
MODE = 1  # 1: train - 2: test

data = load_cifar10_data()
np.random.shuffle(data)
Y_channel = data[:, 0, :].reshape(50000, 32, 32, 1)
UV_channel = data[:, 1:, :].reshape(50000, 32, 32, 2)


def exact_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))


def create_conv(filters, kernel_size, inputs, name=None, bn=True, padding='same', activation='relu'):
    conv = Conv2D(filters, kernel_size, padding=padding, kernel_initializer='he_normal', name=name)(inputs)

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
    pool1 = MaxPool2D((2, 2))(conv1)

    conv2 = create_conv(128, (3, 3), pool1, 'conv2_1', activation='leakyrelu')
    conv2 = create_conv(128, (3, 3), conv2, 'conv2_2', activation='leakyrelu')
    pool2 = MaxPool2D((2, 2))(conv2)

    conv3 = create_conv(256, (3, 3), pool2, 'conv3_1', activation='leakyrelu')
    conv3 = create_conv(256, (3, 3), conv3, 'conv3_2', activation='leakyrelu')
    pool3 = MaxPool2D((2, 2))(conv3)

    conv4 = create_conv(512, (3, 3), pool3, 'conv4_1', activation='leakyrelu')
    conv4 = create_conv(512, (3, 3), conv4, 'conv4_2', activation='leakyrelu')
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

    up9 = create_conv(64, (2, 2), UpSampling2D((2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = create_conv(64, (3, 3), merge9, 'conv9_1', activation='relu')
    conv9 = create_conv(64, (3, 3), conv9, 'conv9_2', activation='relu')
    conv9 = Conv2D(2, (1, 1), padding='same', name='conv9_3')(conv9)

    model = Model(inputs=inputs, outputs=conv9)
    model.compile(optimizer=Adam(LEARNING_RATE),
                  loss='mean_squared_error',
                  metrics=['accuracy', exact_acc, metrics.mse, metrics.mae])

    return model


model = create_model()

if os.path.exists(WEIGHTS):
    model.load_weights(WEIGHTS)

if MODE == 1:
    model_checkpoint = ModelCheckpoint(
        filepath=WEIGHTS,
        monitor='loss',
        verbose=1,
        save_best_only=True)

    reduce_lr = ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=10)

    model.fit(
        Y_channel,
        UV_channel,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        validation_split=0.1,
        callbacks=[model_checkpoint, reduce_lr])

elif MODE == 2:
    for i in range(45000, 50000):
        y = Y_channel[i].T
        yuv_original = np.r_[(y, UV_channel[i].T[:1], UV_channel[i].T[1:])]
        uv_pred = np.array(model.predict(Y_channel[i][None, :, :, :]))[0]
        yuv_pred = np.r_[(y, uv_pred.T[:1], uv_pred.T[1:])]
        show_yuv(yuv_original, yuv_pred)
