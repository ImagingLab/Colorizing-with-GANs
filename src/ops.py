import numpy as np
import tensorflow as tf


def conv2d(inputs, filters, name, kernel_size=4, strides=2, bnorm=True, activation=tf.nn.relu):
    """
    Creates a conv2D block
    """
    initializer = tf.random_normal_initializer(0, 0.02)
    res = tf.layers.conv2d(
        name=name,
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        activation=activation,
        kernel_initializer=initializer)

    if bnorm:
        res = tf.layers.batch_normalization(inputs=res, training=True)

    return res


def conv2d_transpose(inputs, filters, name, kernel_size=4, strides=2, bnorm=True, activation=tf.nn.relu):
    """
    Creates a conv2D-transpose block
    """
    initializer = tf.random_normal_initializer(0, 0.02)
    res = tf.layers.conv2d_transpose(
        name=name,
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        activation=activation,
        kernel_initializer=initializer)

    if bnorm:
        res = tf.layers.batch_normalization(inputs=res, training=True)

    return res


def pixelwise_accuracy(y_true, y_pred):
    """
    Measures the accuracy of the colorization process by comparing pixels
    """
    return tf.reduce_mean(tf.equal(tf.round(y_true), tf.round(y_pred)))


def kernel(filters, strides=2, dropout=False):
    return {
        'filters': filters,
        'strides': strides,
        'dropout': dropout
    }
