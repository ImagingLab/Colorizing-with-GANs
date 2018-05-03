import numpy as np
import tensorflow as tf
from scipy.misc import imread

num_batches = 5
batch_size = 2

class Generator:
    def __init__(self):
        pass
    
    def __iter__(self):
        while True:
            x = np.random.normal()
            y = 3 + 5 * x
            x, y = np.asarray([x, y], np.float32)
            yield x, y

generator_state1 = Generator()

dataset = tf.data.Dataset.from_generator(lambda: generator_state1, (tf.float32, tf.float32))

session = tf.Session()
prefetched = dataset.prefetch(3 * batch_size)
batches = prefetched.batch(batch_size)
iterator = batches.make_one_shot_iterator()

x, y = iterator.get_next()
w = tf.Variable([0, 0], dtype=tf.float32)
prediction = w[0] + w[1] * x
loss = tf.losses.mean_squared_error(y, prediction)
optimizer = tf.train.AdamOptimizer(0.1)
train_op = optimizer.minimize(loss)
init_op = tf.global_variables_initializer()

session = tf.Session()
session.run(init_op)

for _ in range(num_batches):
    _, _loss = session.run([train_op, loss])

a = 1