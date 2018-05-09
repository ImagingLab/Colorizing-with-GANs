
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from abc import abstractmethod
from .networks import Generator, Discriminator
from .dataset import CIFAR10_DATASET, PLACES365_DATASET
from .dataset import Places365Dataset, Cifar10Dataset
from .ops import pixelwise_accuracy, preprocess, postprocess
from .ops import COLORSPACE_RGB, COLORSPACE_LAB
from .utils import stitch_images


class BaseModel:
    def __init__(self, sess, options):
        self.sess = sess
        self.options = options
        self.name = 'CGAN_' + options.dataset
        self.checkpoints_dir = os.path.join(options.checkpoints_path, options.dataset)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.dataset_train = self.create_dataset(True)
        self.dataset_test = self.create_dataset(False)
        self.dataset_test_generator = self.dataset_test.generator(options.samples_size)
        self.iteration = 0
        self.epoch = 0
        self.is_built = False

    def train(self):
        self.build()

        start_time = time.time()
        total = len(self.dataset_train)

        for epoch in range(self.options.epochs):
            self.epoch = epoch
            self.iteration = 0
            generator = self.dataset_train.generator(self.options.batch_size)

            for input_rgb in generator:
                self.iteration += 1
                feed_dic = {self.input_rgb: input_rgb}

                self.sess.run([self.dis_train], feed_dict=feed_dic)
                self.sess.run([self.gen_train, self.accuracy], feed_dict=feed_dic)
                self.sess.run([self.gen_train, self.accuracy], feed_dict=feed_dic)

                errD_fake = self.dis_loss_fake.eval(feed_dict=feed_dic)
                errD_real = self.dis_loss_real.eval(feed_dict=feed_dic)
                errD = errD_fake + errD_real

                errG_l1 = self.gen_loss_l1.eval(feed_dict=feed_dic)
                errG_gan = self.gen_loss_gan.eval(feed_dict=feed_dic)
                errG = errG_l1 + errG_gan

                acc = self.accuracy.eval(feed_dict=feed_dic)

                print("epoch: %d iteration: %4d [%4d/%4d] time: %4.4f, D loss: %.4f (fake: %.4f - real: %.4f), G loss: %.4f (L1: %.4f, gan: %.4f), accuracy: %.4f" % (
                    epoch + 1,
                    self.iteration,
                    self.iteration * self.options.batch_size,
                    total,
                    time.time() - start_time,
                    errD,
                    errD_fake,
                    errD_real,
                    errG,
                    errG_l1,
                    errG_gan,
                    acc)
                )


                # log model at checkpoints
                if self.iteration % self.options.log_interval == 0 and self.iteration > 0:
                    self.sample(show=False)


                # save model at checkpoints
                if self.iteration % self.options.save_interval == 0 and self.iteration > 0:
                    self.save()

            print('\n')

    def sample(self, show=True):
        self.build()

        input_rgb = next(self.dataset_test_generator)
        feed_dic = {self.input_rgb: input_rgb}

        fake_image, input_gray = self.sess.run([self.sampler, self.input_gray], feed_dict=feed_dic)
        fake_image = postprocess(fake_image, colorspace_in=self.options.color_space, colorspace_out=COLORSPACE_RGB)
        img = stitch_images(input_gray, input_rgb, fake_image.eval())

        if not os.path.exists(self.options.samples_path):
            os.makedirs(self.options.samples_path)

        sample = self.options.dataset + "_E" + str(self.epoch).zfill(2) + "_I" + str(self.iteration).zfill(5) + ".png"

        if show:
            plt.imshow(np.array(img))
            plt.axis('off')
            plt.show()
        else:
            print('Saving sample ' + sample)
            img.save(os.path.join(self.options.samples_path, sample))

    def test(self):
        while True:
            self.sample()

    def build(self):
        if self.is_built:
            return

        self.is_built = True

        # create models
        gen = self.create_generator()
        dis = self.create_discriminator()
        sce = tf.nn.sigmoid_cross_entropy_with_logits

        input_shape = self.get_input_shape()

        self.input_rgb = tf.placeholder(tf.float32, shape=(None, input_shape[0], input_shape[1], input_shape[2]), name='input_rgb')
        self.input_gray = tf.image.rgb_to_grayscale(self.input_rgb)
        self.input_color = preprocess(self.input_rgb, colorspace_in=COLORSPACE_RGB, colorspace_out=self.options.color_space)

        self.dis = dis.create(inputs=tf.concat([self.input_gray, self.input_color], 3))
        self.gen = gen.create(inputs=self.input_gray)
        self.gan = dis.create(inputs=tf.concat([self.input_gray, self.gen], 3), reuse_variables=True)
        self.sampler = gen.create(inputs=self.input_gray, reuse_variables=True)


        self.dis_loss_real = tf.reduce_mean(sce(logits=self.dis, labels=tf.ones_like(self.dis)))
        self.dis_loss_fake = tf.reduce_mean(sce(logits=self.gan, labels=tf.zeros_like(self.gan)))
        self.dis_loss = self.dis_loss_real + self.dis_loss_fake

        self.gen_loss_gan = tf.reduce_mean(sce(logits=self.gan, labels=tf.ones_like(self.gan)))
        self.gen_loss_l1 = tf.reduce_mean(tf.abs(self.input_color - self.gen)) * self.options.l1_weight
        self.gen_loss = self.gen_loss_gan + self.gen_loss_l1

        self.accuracy = pixelwise_accuracy(self.input_color, self.gen, self.options.color_space)


        # generator optimizaer
        self.gen_train = tf.train.AdamOptimizer(
            learning_rate=self.options.lr,
            beta1=self.options.beta1
        ).minimize(self.gen_loss, var_list=gen.var_list)

        # discriminator optimizaer
        self.dis_train = tf.train.AdamOptimizer(
            learning_rate=self.options.lr,
            beta1=self.options.beta1
        ).minimize(self.dis_loss, var_list=dis.var_list, global_step=self.global_step)

        self.saver = tf.train.Saver()

    def load(self):
        ckpt = tf.train.get_checkpoint_state(self.checkpoints_dir)

        if ckpt is not None:
            print('loading model...\n')
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.checkpoints_dir, ckpt_name))
            return True

        return False

    def save(self):
        print('saving model...\n')
        self.saver.save(self.sess, os.path.join(self.checkpoints_dir, self.name), global_step=self.global_step)

    @abstractmethod
    def get_input_shape(self):
        raise NotImplementedError

    @abstractmethod
    def create_generator(self):
        raise NotImplementedError

    @abstractmethod
    def create_discriminator(self):
        raise NotImplementedError

    @abstractmethod
    def create_dataset(self, training):
        raise NotImplementedError


class Cifar10Model(BaseModel):
    def __init__(self, sess, options):
        super(Cifar10Model, self).__init__(sess, options)

    def get_input_shape(self):
        return (32, 32, 3)

    def create_generator(self):
        kernels_gen_encoder = [
            (64, 1, 0),     # [batch, 32, 32, ch] => [batch, 32, 32, 64]
            (128, 2, 0),    # [batch, 32, 32, 64] => [batch, 16, 16, 128]
            (256, 2, 0),    # [batch, 16, 16, 128] => [batch, 8, 8, 256]
            (512, 2, 0),    # [batch, 8, 8, 256] => [batch, 4, 4, 512]
            (512, 2, 0)     # [batch, 4, 4, 512] => [batch, 2, 2, 512]
        ]

        kernels_gen_decoder = [
            (512, 2, 0.5),  # [batch, 2, 2, 512] => [batch, 4, 4, 512]
            (256, 2, 0),    # [batch, 4, 4, 512] => [batch, 8, 8, 256]
            (128, 2, 0),    # [batch, 8, 8, 256] => [batch, 16, 16, 128]
            (64, 2, 0)      # [batch, 16, 16, 128] => [batch, 32, 32, 512]
        ]

        return Generator('gen', kernels_gen_encoder, kernels_gen_decoder)

    def create_discriminator(self):
        kernels_dis = [
            (64, 2, 0),     # [batch, 32, 32, ch] => [batch, 16, 16, 64]
            (128, 2, 0),    # [batch, 16, 16, 64] => [batch, 8, 8, 128]
            (256, 2, 0),    # [batch, 8, 8, 128] => [batch, 4, 4, 256]
            (512, 1, 0)     # [batch, 4, 4, 256] => [batch, 4, 4, 512]
        ]

        return Discriminator('dis', kernels_dis)

    def create_dataset(self, training=True):
        return Cifar10Dataset(
            path=self.options.dataset_path,
            training=training,
            augment=self.options.augment)


class Places365Model(BaseModel):
    def __init__(self, sess, options):
        super(Places365Model, self).__init__(sess, options)

    def get_input_shape(self):
        return (256, 256, 3)

    def create_generator(self):
        kernels_gen_encoder = [
            (64, 1, 0),     # [batch, 256, 256, ch] => [batch, 256, 256, 64]
            (64, 2, 0),     # [batch, 256, 256, 64] => [batch, 128, 128, 64]
            (128, 2, 0),    # [batch, 128, 128, 64] => [batch, 64, 64, 128]
            (256, 2, 0),    # [batch, 64, 64, 128] => [batch, 32, 32, 256]
            (512, 2, 0),    # [batch, 32, 32, 256] => [batch, 16, 16, 512]
            (512, 2, 0),    # [batch, 16, 16, 512] => [batch, 8, 8, 512]
            (512, 2, 0),    # [batch, 8, 8, 512] => [batch, 4, 4, 512]
            (512, 2, 0)     # [batch, 4, 4, 512] => [batch, 2, 2, 512]
        ]

        kernels_gen_decoder = [
            (512, 2, 0.5),  # [batch, 2, 2, 512] => [batch, 4, 4, 512]
            (512, 2, 0.5),  # [batch, 4, 4, 512] => [batch, 8, 8, 512]
            (512, 2, 0.5),  # [batch, 8, 8, 512] => [batch, 16, 16, 512]
            (256, 2, 0),    # [batch, 16, 16, 512] => [batch, 32, 32, 256]
            (128, 2, 0),    # [batch, 32, 32, 256] => [batch, 64, 64, 128]
            (64, 2, 0),     # [batch, 64, 64, 128] => [batch, 128, 128, 64]
            (64, 2, 0)      # [batch, 128, 128, 64] => [batch, 256, 256, 64]
        ]

        return Generator('gen', kernels_gen_encoder, kernels_gen_decoder)

    def create_discriminator(self):
        kernels_dis = [
            (64, 2, 0),     # [batch, 256, 256, ch] => [batch, 128, 128, 64]
            (128, 2, 0),    # [batch, 128, 128, 64] => [batch, 64, 64, 128]
            (256, 2, 0),    # [batch, 64, 64, 128] => [batch, 32, 32, 256]
            (512, 2, 0),    # [batch, 32, 32, 256] => [batch, 16, 16, 512]
            (512, 2, 0),    # [batch, 16, 16, 512] => [batch, 8, 8, 512]
            (512, 2, 0)     # [batch, 8, 8, 512] => [batch, 4, 4, 512]
        ]

        return Discriminator('dis', kernels_dis)

    def create_dataset(self, training=True):
        return Places365Dataset(
            path=self.options.dataset_path,
            training=training,
            augment=self.options.augment)


def model_factory(sess, options):
    if options.dataset == CIFAR10_DATASET:
        model = Cifar10Model(sess, options)

    elif options.dataset == PLACES365_DATASET:
        model = Places365Model(sess, options)

    if not os.path.exists(model.checkpoints_dir):
        os.makedirs(model.checkpoints_dir)

    model.build()
    model.load()
    return model
