from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf

from abc import abstractmethod
from .dataset import *
from .ops import pixelwise_accuracy
from .networks import Generator, Discriminator
from .dataset import Places365Dataset, Cifar10Dataset


class BaseModel:
    def __init__(self, sess, options):
        self.sess = sess
        self.options = options
        self.name = 'COLGAN_' + options.dataset
        self.saver = tf.train.Saver()
        self.path = os.path.join(options.checkpoint_path, self.name)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.dataset_train = self.create_dataset(True)
        self.dataset_test = self.create_dataset(False)

    def train(self):
        start_time = time.time()
        total = len(self.dataset_train)

        for epoch in range(self.options.epochs):
            batch_counter = 0
            generator = self.dataset_train.generator(self.options.batch_size)

            for input_color in generator:
                batch_counter += 1
                input_gray = rgb2gray(input_color)
                input_color = preprocess(input_color, self.options.color_space)

                gen_feed_dic = {self.input_color: input_color}
                dis_feed_dic = {self.input_color: input_color, self.input_gray: input_gray}

                self.sess.run([self.dis_train, self.accuracy], feed_dict=dis_feed_dic)
                self.sess.run([self.gen_train, self.accuracy], feed_dict=gen_feed_dic)
                self.sess.run([self.gen_train, self.accuracy], feed_dict=gen_feed_dic)

                errD_fake = self.dis_loss_fake.eval(feed_dict=gen_feed_dic)
                errD_real = self.dis_loss_real.eval(feed_dict=dis_feed_dic)
                errG_l1 = self.gen_loss_l1.eval(feed_dict=gen_feed_dic)
                errG_gan = self.gen_loss_gan.eval(feed_dict=gen_feed_dic)
                acc = self.accuracy.eval(feed_dict=gen_feed_dic)

                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, D loss: %.8f, G total loss: %.8f, G L1: %.8f, G gan: %.8f, accuracy: %.8f" % (
                    epoch, 
                    batch_counter * self.options.batch_size,
                    total, 
                    time.time() - start_time, 
                    errD_fake + errD_real, 
                    errG_l1 + errG_gan, 
                    errG_l1, 
                    errG_gan,
                    acc)
                )


                # log model at checkpoints
                if batch_counter % self.options.log_interval == 0 and batch_counter > 0:
                    self.sample()


                # save model at checkpoints
                if batch_counter % self.options.save_interval == 0 and batch_counter > 0:
                    self.save()

    def test(self):
        generator = self.dataset_test.generator(1)
        for real_image in generator:
            input_gray = rgb2gray(real_image)
            input_color = preprocess(real_image, self.options.color_space)
            fake_image = self.sess.run(self.sampler, feed_dict={self.input_color: input_color, self.input_gray: input_gray})
            fake_image = postprocess(fake_image, self.options.color_space)
            imshow(input_color, fake_image, self.options.color_space)

    def sample(self):
        sample_size = 16
        generator = self.dataset_test.generator(sample_size)
        real_images = next(generator)
        inputs_gray = rgb2gray(real_images)
        inputs_color = preprocess(real_images, self.options.color_space)
        fake_images = self.sess.run(self.sampler, feed_dict={self.input_color: inputs_color, self.input_gray: inputs_gray})
        fake_images = postprocess(fake_images, self.options.color_space)
        # save images


    def build(self):
        # create models
        gen = self.create_generator()
        dis = self.create_discriminator()
        sce = tf.nn.sigmoid_cross_entropy_with_logits

        self.input_gray = tf.placeholder(tf.float32, shape=(None, None, None, 1), name='input_gray')
        self.input_color = tf.placeholder(tf.float32, shape=(None, None, None, 3), name='input_color')

        self.gen = gen.create(inputs=self.input_gray)
        self.dis = dis.create(inputs=tf.concat([self.input_gray, self.input_color], 3))
        self.gan = dis.create(inputs=tf.concat([self.input_gray, self.gen], 3), reuse_variables=True)
        self.sampler = gen.create(inputs=self.input_gray, reuse_variables=True)


        self.dis_loss_real = tf.reduce_mean(sce(logits=self.dis, labels=tf.ones_like(self.dis) * 0.9))
        self.dis_loss_fake = tf.reduce_mean(sce(logits=self.gen, labels=tf.zeros_like(self.gen)))
        self.dis_loss = self.dis_loss_real + self.dis_loss_fake

        self.gen_loss_gan = tf.reduce_mean(sce(logits=self.gen, labels=tf.ones_like(self.gen)))
        self.gen_loss_l1 = tf.reduce_mean(tf.abs(self.input_color - self.gen))
        self.gen_loss = self.gen_loss_gan + self.options.l1_weight * self.gen_loss_l1

        self.accuracy = pixelwise_accuracy(self.input_color, self.gen)


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

    if not os.path.exists(model.path):
        os.makedirs(model.path)
    else:
        model.load()

    return model
