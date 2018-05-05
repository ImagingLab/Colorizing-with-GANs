from __future__ import print_function

import numpy as np
import tensorflow as tf

from .dataset import *
from .ops import pixelwise_accuracy, kernel
from .networks import Generator, Discriminator


class ColorizationModel:
    def __init__(self, sess, options):
        self.sess = sess
        self.options = options
        self.name = 'COLGAN_' + options.dataset
        self.saver = tf.train.Saver()
        self.path = os.path.join(options.checkpoint_path, self.name)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

    def train(self):
        pass

    def test(self):
        pass

    def build(self):
        gen, dis, gan = self.create_models()
        #tf.concat([color_inputs, grayscale_inputs], axis=3)

    def create_models(self):

        kernels_gen_encoder = []
        kernels_gen_decoder = []
        kernels_dis = []

        # load kernels
        if self.options.dataset == CIFAR10_DATASET:
            kernels_gen_encoder = [
                # encoder_1: [batch, 32, 32, ch] => [batch, 32, 32, 64]
                kernel(64, 1, 0),
                # encoder_2: [batch, 32, 32, 64] => [batch, 16, 16, 128]
                kernel(128, 2, 0),
                # encoder_3: [batch, 16, 16, 128] => [batch, 8, 8, 256]
                kernel(256, 2, 0),
                # encoder_4: [batch, 8, 8, 256] => [batch, 4, 4, 512]
                kernel(512, 2, 0),
                # encoder_5: [batch, 4, 4, 512] => [batch, 2, 2, 512]
                kernel(512, 2, 0)
            ]

            kernels_gen_decoder = [
                # decoder_1: [batch, 2, 2, 512] => [batch, 4, 4, 512]
                kernel(512, 2, 0.5),
                # decoder_2: [batch, 4, 4, 512] => [batch, 8, 8, 256]
                kernel(256, 2, 0),
                # decoder_3: [batch, 8, 8, 256] => [batch, 16, 16, 128]
                kernel(128, 2, 0),
                # decoder_4: [batch, 16, 16, 128] => [batch, 32, 32, 512]
                kernel(64, 2, 0)
            ]

            kernels_dis = [
                # layer_1: [batch, 32, 32, ch] => [batch, 16, 16, 64]
                kernel(64, 2, 0),
                # layer_2: [batch, 16, 16, 64] => [batch, 8, 8, 128]
                kernel(128, 2, 0),
                # layer_3: [batch, 8, 8, 128] => [batch, 4, 4, 256]
                kernel(256, 2, 0),
                # layer_4: [batch, 4, 4, 256] => [batch, 4, 4, 512]
                kernel(512, 1, 0)
            ]

        elif self.options.dataset == PLACES365_DATASET:
            kernels_gen_encoder = [
                # encoder_1: [batch, 256, 256, ch] => [batch, 256, 256, 64]
                kernel(64, 1, 0),
                # encoder_2: [batch, 256, 256, 64] => [batch, 128, 128, 64]
                kernel(64, 2, 0),
                # encoder_3: [batch, 128, 128, 64] => [batch, 64, 64, 128]
                kernel(128, 2, 0),
                # encoder_4: [batch, 64, 64, 128] => [batch, 32, 32, 256]
                kernel(256, 2, 0),
                # encoder_5: [batch, 32, 32, 256] => [batch, 16, 16, 512]
                kernel(512, 2, 0),
                # encoder_6: [batch, 16, 16, 512] => [batch, 8, 8, 512]
                kernel(512, 2, 0),
                # encoder_7: [batch, 8, 8, 512] => [batch, 4, 4, 512]
                kernel(512, 2, 0),
                # encoder_8: [batch, 4, 4, 512] => [batch, 2, 2, 512]
                kernel(512, 2, 0)
            ]

            kernels_gen_decoder = [
                # decoder_1: [batch, 2, 2, 512] => [batch, 4, 4, 512]
                kernel(512, 2, 0.5),
                # decoder_2: [batch, 4, 4, 512] => [batch, 8, 8, 512]
                kernel(512, 2, 0.5),
                # decoder_3: [batch, 8, 8, 512] => [batch, 16, 16, 512]
                kernel(512, 2, 0.5),
                # decoder_4: [batch, 16, 16, 512] => [batch, 32, 32, 256]
                kernel(256, 2, 0),
                # decoder_5: [batch, 32, 32, 256] => [batch, 64, 64, 128]
                kernel(128, 2, 0),
                # decoder_6: [batch, 64, 64, 128] => [batch, 128, 128, 64]
                kernel(64, 2, 0),
                # decoder_7: [batch, 128, 128, 64] => [batch, 256, 256, 64]
                kernel(64, 2, 0)
            ]

            kernels_dis = [
                # layer_1: [batch, 256, 256, ch] => [batch, 128, 128, 64]
                kernel(64, 2, 0),
                # layer_2: [batch, 128, 128, 64] => [batch, 64, 64, 128]
                kernel(128, 2, 0),
                # layer_3: [batch, 64, 64, 128] => [batch, 32, 32, 256]
                kernel(256, 2, 0),
                # layer_4: [batch, 32, 32, 256] => [batch, 16, 16, 512]
                kernel(512, 2, 0),
                # layer_5: [batch, 16, 16, 512] => [batch, 8, 8, 512]
                kernel(512, 2, 0),
                # layer_6: [batch, 8, 8, 512] => [batch, 4, 4, 512]
                kernel(512, 2, 0)
            ]

        # create model factories
        gen = Generator('gen', kernels_gen_encoder, kernels_gen_decoder)
        dis = Discriminator('dis', kernels_dis)
        gan = Discriminator('gan', kernels_dis)

        return gen, dis, gan

    def load(self):
        print('loading model...\n')

        ckpt = tf.train.get_checkpoint_state(self.path)

        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, self.path)
            return True

        else:
            print("failed to find a checkpoint")
            return False

    def save(self):
        print('saving model...\n')
        self.saver.save(self.sess, self.path, global_step=self.global_step)
