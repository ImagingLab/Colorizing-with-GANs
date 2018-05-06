import os
import random
import numpy as np
import tensorflow as tf
from src import ModelOptions, model_factory

# read input arguments
options = ModelOptions().parse()


# initialize random seed
tf.set_random_seed(options.seed)
np.random.seed(options.seed)
random.seed(options.seed)


# create a session environment
with tf.Session() as sess:
    model = model_factory(sess, options)
    sess.run(tf.global_variables_initializer())

    if options.train:
        model.train()
    else:
        model.test()
