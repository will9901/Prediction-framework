'''
Created on Oct 02, 2017

@author: Ying Hu

'''	

import tensorflow as tf

num_units = 128


def cnn(data):

	# convolutional layer
	conv = tf.layers.conv2d(
		inputs = data,
		filters = 64,
		kernel_size = [5,5],
		padding = 'same',
		activation = tf.nn.relu)

	# pooling layer
	pool = tf.layers.max_pooling2d(
		inputs = conv,
		pool_size = [1,3],
		strides = 1)

	# dense layer
	pool_flat = tf.reshape(pool, [-1,64*2])
	dense = tf.layers.dense(inputs = pool_flat,
		units = num_units,
		activation = tf.nn.relu)

	return dense


if __name__ == '__main__':
	data = tf.placeholder(tf.float32, [None, None])
	b = tf.reshape(data,[-1,1,7,1])
	dense = cnn(b)
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	print sess.run(dense, {data: [[1,3,4,3,2,4,2,1,3,5,5,2,1,24],[5,2,7,89,1,9,0,1,3,2,3,5,6,7]]})
