import numpy as np
import tensorflow as tf
from ops import conv2d, conv2d_transpose, pixelwise_accuracy


class Discriminator(object):
    def __init__(self, name, kernels):
        self.name = name
        self.kernels = kernels
        self.var_list = []

    def __call__(self, input, reuse_variables=None):
        output = input
        with tf.variable_scope(self.name, reuse=reuse_variables) as scope:
            for index, kernel in enumerate(self.kernels):

                # not use batch-norm in the first layer
                bnorm = False if index == 0 else True
                output = conv2d(
                    inputs=output,
                    name='conv' + str(index),
                    filters=kernel['filters'],
                    strides=kernel['strides'],
                    bnorm=bnorm,
                    activation=tf.nn.leaky_relu
                )

                if kernel['dropout']:
                    output = tf.nn.dropout(output, keep_prob=0.5)

            output = tf.reshape(output, [-1, np.prod(output.shape[1:])])
            output = tf.layers.dense(inputs=output, units=1)

            self.var_list = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

            return output


class Generator(object):
    def __init__(self, name, encoder_kernels, decoder_kernels, output_channels=3):
        self.name = name
        self.encoder_kernels = encoder_kernels
        self.decoder_kernels = decoder_kernels
        self.output_channels = output_channels
        self.var_list = []

    def __call__(self, input, reuse_variables=None):
        output = input

        with tf.variable_scope(self.name, reuse=reuse_variables) as scope:

            layers = []

            # encoder branch
            for index, kernel in enumerate(self.encoder_kernels):

                output = conv2d(
                    inputs=output,
                    name='conv' + str(index),
                    filters=kernel['filters'],
                    strides=kernel['strides'],
                    activation=tf.nn.leaky_relu
                )

                layers.append(output)

                if kernel['dropout']:
                    output = tf.nn.dropout(output, keep_prob=0.5)

            # decoder branch
            for index, kernel in enumerate(self.decoder_kernels):

                if index > 0:
                    output = tf.concat([output, layers[len(layers) - index - 1]], axis=3)

                output = conv2d_transpose(
                    inputs=output,
                    name='deconv' + str(index),
                    filters=kernel['filters'],
                    strides=kernel['strides'],
                    activation=tf.nn.relu
                )

                if kernel['dropout']:
                    output = tf.nn.dropout(output, keep_prob=0.5)

            output = conv2d(
                inputs=output,
                name='conv_last',
                filters=self.output_channels,
                bnorm=False,
                activation=tf.nn.tanh
            )

            self.var_list = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

            return output
