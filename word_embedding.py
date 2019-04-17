'''

These codes work for embedding hashtag.
The output embedding is a big vector containing the vector of each word of a hashtag.
The vector of each word has 200 dimensional elements. So if a hashtag has n words, then
its vector representation has n*200 dimensional elements.
Note that each hashtag has only one vector of representation.
'''




# returns a list 
def make_embedding(hashtag):
	output = []
	for line in open('dataset_www2018_hashtagtovector.txt'):
		line = line.strip().split('\t')
		hashtag_temp = line[0]
		vector_temp = line[1]
		
		if hashtag == hashtag_temp:
			num_list = vector_temp[1:len(vector_temp)-1].strip().split(',')
			for item in num_list:
				item = item.strip().strip(' ')
				if item !='0.0':
					output.append(float(item[1:len(item)-1]))
				else:
					output.append(float(item))
			break


	return output




if __name__ == '__main__':
	output = make_embedding('funnymartinmoment')

	print (output)
	

