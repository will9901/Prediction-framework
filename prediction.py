# -*- coding: utf-8 -*-
# 作者         ：HuangJianyi
'''

All are linked together here.
input-->embeddings-->lstm-->predictor-->using gredient descent
'''

import numpy as np
import tensorflow as tf
import predictor
import lstm
import cnn
import length_and_count_embedding


time_steps = 10

num_epochs = 5
batch_size = 100
num_iterations = 700/batch_size 
learning_rate = 0.03
num_classes = 1

lstm_max_length = time_steps
num_lstm_features = 3206

htg_acpd_dict = {}


def get_lstm_data(hashtag_list):

	d3_list = []
	sequence_length_vector = []

	for item in hashtag_list:
		active_period = htg_acpd_dict[hashtag]
		if time_steps >= active_period:
			sequence_length_vector.append(active_period)
		else:
			sequence_length_vector.append(time_steps)
		d2_list = []
		for i in range(time_steps):
			d1_list = []
			path = './lstm_input/lstm_input_' + str(i+1) +'.txt'
			for line in open(path):
				line = line.strip().split('\t')
				hashtag_temp = line[0]
				if hashtag_temp == item:
					l = line[1][1:len(line[1])-1].strip().split(',')
					for k in l:
						d1_list.append(float(k))
					break
			d2_list.append(d1_list)
		d3_list.append(d2_list)

	return sequence_length_vector, d3_list




def get_cnn_data(hashag_list):
	d2_list = []
	for item in hashag_list:
		d1_list = []
		for line in open('dataset_www2018_hashtagtovector.txt'):
			line = line.strip().split('\t')
			hashtag_temp = line[0]
			if hashtag_temp == item:
				l = line[1][1:len(line[1])-1].strip().split(',')
				for k in l:
					d1_list.append(float(k))
				break
		d2_list.append(d1_list)
	return d2_list



def get_length_and_count_embeddings(hashag_list):


def get_target_list(hashag_list):



def get_trainingset_hashtag():
	l = []
	i = 0
	for line in open('dataset_www2018_rand.txt'):
		i = i+1
		line = line.strip().split('\t')
		hashtag = line[0]
		l.append(hashtag)

		if i==1000:
			break

	return l



def get_testset_hashtag():
	l = []
	i = 0
	for line in open('dataset_www2018_rand.txt'):
		i = i+1
		if i>1000:
			line = line.strip().split('\t')
			hashtag = line[0]
			l.append(hashtag)

	return l


def overall_prediction(lstm_output, cnn_output, length_and_count_embeddings):
	# joint vector part
	joint_vector = []


	output = predictor.predictor(joint_vector)

	return output



def cost(prediction, target):
	return tf.reduce_mean(tf.square(prediction - target))



def optimizer(cost):
	return tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


def main():

	for line in open('dataset_www2018_rand.txt'):
		line = line.strip().split('\t')
		hashtag = line[0]
		active_period = int(line[1])
		htg_acpd_dict[hashtag] = active_period


	hashtag_trainingset = get_trainingset_hashtag()
	print '!!!\nhashtags of the trainingset have been picked out.\n!!!'
	
	target = tf.placeholder(tf.float32, [None, num_classes])

	lstm_data = tf.placeholder(tf.float32, [None, lstm_max_length, num_lstm_features])	
	lstm_sequence_length_vector = tf.placeholder(tf.int32, [None])
	lstm_dropout = tf.placeholder(tf.float32)
	lstm_output = lstm.lstm(lstm_data, lstm_sequence_length_vector, lstm_dropout)


	cnn_data = tf.placeholder(tf.float32, [None, None])
	reshape_cnn_data = tf.reshape(data,[-1,1,100,1])
	cnn_output = cnn.cnn(reshape_cnn_data)

	#getcountlengthembeddings part
	length_and_count
	length_and_count_embeddings = get_length_and_count_embeddings()

	prediction_output = overall_prediction(lstm_output, cnn_output, length_and_count_embeddings)
	cost_output = cost(prediction_output,target)
	optimizer_output = optimizer(cost)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())



	for epoch in range(num_epochs):
		np.random.shuffle(hashtag_trainingset)
		mse_sum = 0
		for i in range(num_iterations):


if __name__ == '__main__':
	main()
