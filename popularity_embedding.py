# -*- coding: utf-8 -*-
# 作者         ：HuangJianyi
'''

These codes work for embedding popularity information.
Given a hashtag, the output embedding contains the cumulative popularity
of a given time step.
'''


# returns a list containing one element
def make_embedding(hashtag, time_step):
	output = [-1]
	for line in open('dataset_www2018_cumupopularityevolution.txt'):
		line = line.strip().split('\t')
		hashtag_temp = line[0]
		hour_temp = int(line[1])
		cumupopularity_temp = int(line[2])

		if hashtag == hashtag_temp and hour_temp == time_step:
			output = [cumupopularity_temp]
			break
	return output


	

if __name__ == '__main__':
	output = make_embedding('ryanleslieonbetnow.', 1)

	print (output)
