import os
import random
import numpy as np
import tensorflow as tf
from src import *

# read input arguments
options = ModelOptions().parse()


# initialize random seed
tf.set_random_seed(options.seed)
np.random.seed(options.seed)
random.seed(options.seed)


# create a session environment
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    model = model_factory(sess, options)

    if options.train:
        model.train()
    else:
        model.test()
