'''
Created on Sep 14, 2017

@author: Ying Hu

key reference: https://danijar.com/introduction-to-recurrent-networks-in-tensorflow/
 			   https://danijar.com/variable-sequence-lengths-in-tensorflow/
'''

import tensorflow as tf


num_units = 200
num_layers = 3



# LSTM cell
def lstm_cell(dropout):
	cell = tf.contrib.rnn.LSTMCell(num_units, reuse=tf.get_variable_scope().reuse)
	return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1 - dropout)


def last_relevant(lstm_output, sequence_length_vector):
	'''
	lstm_output contains the output of every time step, but we only care about the output
	of last relevant time step.
	'''
	batch_size = tf.shape(lstm_output)[0]
	max_length = tf.shape(lstm_output)[1]
	out_size = int(lstm_output.get_shape()[2])
	index = tf.range(0, batch_size) * max_length + (sequence_length_vector - 1)
	flat = tf.reshape(lstm_output, [-1, out_size])
	relevant = tf.gather(flat, index)
	return relevant


def lstm(data, sequence_length_vector, dropout):
	'''
	data: a batch of embeddings of all examples.
	seqence_length_vector: each example has differenct time steps. seqence_length_vector is a vector
	of which each element is the number of time steps of each example.
	dropout: dropout
	This function returns h(t). 
	'''
	# recurrent network
	network = tf.contrib.rnn.MultiRNNCell([lstm_cell(dropout) for _ in range(num_layers)])
	lstm_output, _ = tf.nn.dynamic_rnn(network, data, dtype=tf.float32, sequence_length=sequence_length_vector)

	
	last_output = last_relevant(lstm_output,sequence_length_vector)

	return last_output
