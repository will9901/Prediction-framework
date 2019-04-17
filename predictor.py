# -*- coding: utf-8 -*-
# 作者         ：HuangJianyi
'''

The output of lstm should be fed into a predictor. We choose NN as our predictor.
This NN has three layers. Each of the first two layers has 100 neurons and uses ReLu 
as the activation fuction.The last layer has only 1 neuron without using 
an activation function.
'''


import tensorflow as tf

num_units_layer1 = 100 
num_units_layer2 = 100
num_units_output_layer = 1


# create weights and biases for a nn layer
def weights_and_biases(in_size, out_size):
    weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
    bias = tf.constant(0.1, shape=[out_size])
    return tf.Variable(weight), tf.Variable(bias)


def predictor(data):
	# data: the input of the predictor. the output of lstm.
	in_size_layer1 = tf.shape(data)[1]
	weights_layer1, biases_layer1 = weights_and_biases(in_size_layer1, num_units_layer1)

	weights_layer2, biases_layer2 = weights_and_biases(num_units_layer1, num_units_layer2)

	weights_output_layer, biases_output_layer = weights_and_biases(num_units_layer2, num_units_output_layer)

	layer1 = tf.add(tf.matmul(data,weights_layer1),biases_layer1)
	layer1 = tf.nn.relu(layer1)

	layer2 = tf.add(tf.matmul(layer1,weights_layer2),biases_layer2)
	layer2 = tf.nn.relu(layer2)

	output = tf.add(tf.matmul(layer2,weights_output_layer),biases_output_layer)

	return output



