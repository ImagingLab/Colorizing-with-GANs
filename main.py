import os
import random
import numpy as np
import tensorflow as tf
from src import ModelOptions, model_factory

# read input arguments
options = ModelOptions().parse()


tf.reset_default_graph()

# initialize random seed
tf.set_random_seed(options.seed)
np.random.seed(options.seed)
random.seed(options.seed)

# create a session environment
with tf.Session() as sess:
    model = model_factory(sess, options)
    sess.run(tf.global_variables_initializer())

    # load model only after global variables initialization
    model.load()

    if options.train:
        model.train()
    else:
        model.evaluate()
        while True:
            model.sample()
