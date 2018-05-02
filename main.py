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


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    model = ColorizationModel(sess, options)

    # loading model
    if not os.path.exists(model.path):
        os.makedirs(model.path)
    else:
        model.load()


    if options.train:
        model.train()
    else:
        model.test()
