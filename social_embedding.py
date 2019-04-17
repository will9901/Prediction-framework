'''

These codes work for embedding social information.
The output embedding includes number of celebrities, maximum of the numbers of followers, 
sum of the numbers of followers, mean of the numbers of followers, and
median of the numbers of followers.
Note that the output embeddings is different given a hashtag and its different time step,
because social information evolves over time.
'''

import numpy as np



# returns a list containing five elements
def make_embedding(hashtag, time_step):
	output = [-1,-1,-1,-1,-1]
	for line in open('dataset_www2018_socialevolution.txt'):
		line = line.strip().split('\t')
		hashtag_temp = line[0]
		hour_temp = int(line[1])
		social_temp = line[2]
		
		if hashtag == hashtag_temp and hour_temp == time_step:
			output = social_temp.strip().split(',')
			break


	return output




if __name__ == '__main__':
	output = make_embedding('a', 3)

	print (output)
	

