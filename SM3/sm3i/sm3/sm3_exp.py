from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import tensorflow_datasets as tfds

from sm3 import sm3

"""
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

def creat_mnist_dataset(data, labels, batch_size):
	def gen():
		for image, label in zip(data, labels):
			yield image, label
	ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.int32), ((28,28 ), ()))
	return ds.repeat().batch(batch_size)

train_dataset = create_mnist_dataset(x_train, y_train, 10)
valid_dataset = create_mnist_dataset(x_test, y_test, 20)
"""

mnist_train = tfds.load(name="mnist", split="train")
assert isinstance(mnist_train, tf.data.Dataset)
print(mnist_train)

mnist = tfds.load("mnist:1.*.*")


"""
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	
	for epoch in range(epochs):
		total_batch = int(len(x_train) / batch_size)
		x_batches = np.array_split(x_train, total_batch)
		y_batches = np.array_split(y_train, total_batch)
		for i in range(total_batch):
			batch_x, batch_y = x_batches[i], y_batches[i]
"""
