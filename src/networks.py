import numpy as np
import tensorflow as tf
from .ops import conv2d, conv2d_transpose, pixelwise_accuracy


class Discriminator(object):
    def __init__(self, name, kernels):
        self.name = name
        self.kernels = kernels
        self.var_list = []

    def create(self, inputs, reuse_variables=None):
        output = inputs
        with tf.variable_scope(self.name, reuse=reuse_variables) as scope:
            for index, kernel in enumerate(self.kernels):

                # not use batch-norm in the first layer
                bnorm = False if index == 0 else True
                name = 'conv' + str(index)
                output = conv2d(
                    inputs=output,
                    name=name,
                    filters=kernel[0],
                    strides=kernel[1],
                    bnorm=bnorm,
                    activation=tf.nn.leaky_relu
                )

                if kernel[2] > 0:
                    output = tf.nn.dropout(output, keep_prob=1 - kernel[2], name='dropout_' + name)

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

    def create(self, inputs, reuse_variables=None):
        output = inputs

        with tf.variable_scope(self.name, reuse=reuse_variables) as scope:

            layers = []

            # encoder branch
            for index, kernel in enumerate(self.encoder_kernels):

                name = 'conv' + str(index)
                output = conv2d(
                    inputs=output,
                    name=name,
                    filters=kernel[0],
                    strides=kernel[1],
                    activation=tf.nn.leaky_relu
                )

                layers.append(output)

                if kernel[2] > 0:
                    output = tf.nn.dropout(output, keep_prob=1 - kernel[2], name='dropout_' + name)

            # decoder branch
            for index, kernel in enumerate(self.decoder_kernels):

                if index > 0:
                    output = tf.concat(
                        [output, layers[len(layers) - index - 1]], axis=3)

                name = 'deconv' + str(index)
                output = conv2d_transpose(
                    inputs=output,
                    name=name,
                    filters=kernel[0],
                    strides=kernel[1],
                    activation=tf.nn.relu
                )

                if kernel[2] > 0:
                    output = tf.nn.dropout(output, keep_prob=0.5, name='dropout_' + name)

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
