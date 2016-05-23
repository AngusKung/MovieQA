import numpy as np
import re

def load_2d_array(path):
	f = open(path,'rb')
	arr = np.loadtxt(f,dtype='float32')
	return arr

def save_2d_array(path,data):
	print "Saving to "+path
	with open(path,'wb') as f:
		np.savetxt(f,data)
	f.close()

def load_3d_array(path):
	f = open(path,'rb')
	firstline = f.readline()
	num=[]
	for s in re.findall("[-+]?\d+[\.]?\d*",firstline):
		num.append(int(s))
	assert len(num)==3
	print tuple(num)
	arr = np.loadtxt(f,dtype='float32')
	f.close()
	arr = arr.reshape(tuple(num))
	return arr

def save_3d_array(path,data):
	print "Saving to "+path
	with open(path,'wb') as f:
		f.write('# Array shape: {0}\n'.format(data.shape))
		for data_slice in data:
			#np.savetxt(f,data_slice,fmt='%.7g')
			np.savetxt(f,data_slice)
			f.write('# New slice\n')
	f.close()

# for gru input
# the Answer is a list of 300 dim vectors
# return each P and Q and Each Answer
def mk_newgru300(data):
	data_num = len(data)
	word_dim = len(data[0][0][0][0])
#	print "This is make gru 300\n"
#	print "Total data: %d\nWord dim:  %d\n" % (data_num, word_dim)

	label_size = len(data[0][1])
	#data[0][0] contains P,Q,A1,A2,A3,A4,A5
	#data[0][1] contains label
	#data[0][0][0] contains P is a list of words
	#data[0][0][0][0] is word's np vector dim=300

	passage = []
	questions =[]
	A1 = []
	A2 = []
	A3 = []
	A4 = []
	A5 = []
	true_ans = np.zeros((data_num, label_size),dtype='float32')
	for i in range(data_num):
		oneline=[]
		oneline = oneline + data[i][0][0]
		passage.append(oneline)
		onequestion=[]
		onequestion = onequestion +data[i][0][1]
		questions.append(onequestion)
		#print "Each len is "+ str(len(data_shuf[i]))
		index = np.nonzero(data[i][1])[0]+2
		index = int(index)
		#checkshape = 1
		#if index.ndim != checkshape:
		#	print "Data %d failed\n" % (i)
		# calculate the answer (average all its words)
		true_ans[i, ] = data[i][1]
		
		oneA1 = []
		oneA1 = oneA1 + data[i][0][2]
		A1.append(oneA1)
		oneA2 = []
		oneA2 = oneA2 + data[i][0][3]
		A2.append(oneA2)
		oneA3 = []
		oneA3 = oneA3 + data[i][0][4]
		A3.append(oneA3)
		oneA4 = []
		oneA4 = oneA4 + data[i][0][5]
		A4.append(oneA4)
		oneA5 = []
		oneA5 = oneA5 + data[i][0][6]
		A5.append(oneA5)


	return passage, questions, A1, A2, A3, A4, A5, true_ans

# for gru input
# the label is a 300 dim vector which is the average of all the words in the answer's sentence
# return each P and Q and Each Answer
def mk_gru300(data):
	data_num = len(data)
	word_dim = len(data[0][0][0][0])
	print "This is make gru 300\n"
	print "Total data: %d\nWord dim:  %d\n" % (data_num, word_dim)

	label_size = len(data[0][1])
	#data[0][0] contains P,Q,A1,A2,A3,A4
	#data[0][1] contains label
	#data[0][0][0] contains P is a list of words
	#data[0][0][0][0] is word's np vector dim=300

	passage = []
	questions =[]
	A1 = np.zeros((data_num, word_dim),dtype='float32')
	A2 = np.zeros((data_num, word_dim),dtype='float32')
	A3 = np.zeros((data_num, word_dim),dtype='float32')
	A4 = np.zeros((data_num, word_dim),dtype='float32')
	true_ans = np.zeros((data_num, label_size),dtype='float32')
	for i in range(data_num):
		oneline=[]
		oneline = oneline +data[i][0][0]
		passage.append(oneline)
		onequestion=[]
		onequestion = onequestion +data[i][0][1]
		questions.append(onequestion)
		#print "Each len is "+ str(len(data_shuf[i]))
		index = np.nonzero(data[i][1])[0]+2
		index = int(index)
		#checkshape = 1
		#if index.ndim != checkshape:
		#	print "Data %d failed\n" % (i)
		# calculate the answer (average all its words)
		true_ans[i, ] = data[i][1]
	
		a1_sum = np.zeros(word_dim,dtype='float32')
		for ans_word in data[i][0][2]:
			a1_sum = a1_sum + ans_word
		A1[i, ] = a1_sum/len(data[i][0][2])

		a2_sum = np.zeros(word_dim,dtype='float32')
		for ans_word in data[i][0][3]:
			a2_sum = a2_sum + ans_word
		A2[i, ] = a2_sum/len(data[i][0][3])

		a3_sum = np.zeros(word_dim,dtype='float32')
		for ans_word in data[i][0][4]:
			a3_sum = a3_sum + ans_word
		A3[i, ] = a3_sum/len(data[i][0][4])

		a4_sum = np.zeros(word_dim,dtype='float32')
		for ans_word in data[i][0][5]:
			a4_sum = a4_sum + ans_word
		A4[i, ] = a4_sum/len(data[i][0][5])

	print "Final len is "+str(len(passage))
	return passage, questions, A1, A2, A3, A4, true_ans

