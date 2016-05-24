import argparse
import numpy as np
import sys
import cPickle
import pdb

"""my .py file"""
from parse import mk_newgru300, save_3d_array, load_3d_array, save_2d_array, load_2d_array
from mybiGRU import oneGRUmodel, twoGRUmodel,MQA_datagen
from mybiGRU import loadMQAPickle,findMaxlen


"""Keras"""
from keras.preprocessing.sequence import pad_sequences


parser = argparse.ArgumentParser()
parser.add_argument("-dim_gru", type=int, default=128)
parser.add_argument("-n_hid_layers", type=int, default=2)
parser.add_argument("-dropout", type=float, default=0)
parser.add_argument("-activation", type=str, default="softplus")
parser.add_argument("-epochs", type=int, default=1000)
parser.add_argument("-model_save_interval", type=int, default=10)
parser.add_argument("-batch_size", type=int, default=32)
parser.add_argument("-lr", type=float, default=0.001)
parser.add_argument("-hop", type=int, default=0)
parser.add_argument("-atten_mode", type=str, choices=['sigmoid','softmax'], default="softmax")
parser.add_argument("-path",type=str,required=True)
parser.add_argument("-split",type=str, choices=['train','val','test'],required=True)
parser.add_argument("-type", type=str, choices=['plot','dvs','subtitle','script'] , required=True)
args = parser.parse_args()

print "Running with args:"
print args

dim_glove = 300
#path = './Pickle/val.plot.lstm.pickle'


print "Loading MovieQA data..."
training_data  = loadMQAPickle(args.path)
passages, questions, A1, A2, A3, A4, A5, true_ans = mk_newgru300(training_data)

maxlen = findMaxlen(A1)
maxlen = findMaxlen(A2,maxlen)
maxlen = findMaxlen(A3,maxlen)
maxlen = findMaxlen(A4,maxlen)
maxlen = findMaxlen(A5,maxlen)
maxlen = findMaxlen(questions,maxlen)
print "MAX_len A&Q  : "+str(maxlen)
maxlen_pass = findMaxlen(passages)
print "MAX_len pass : "+str(maxlen_pass)

passages = pad_sequences(passages, maxlen=maxlen_pass, dtype='float32')
questions = pad_sequences(questions, maxlen=maxlen, dtype='float32')
A1 = pad_sequences(A1, maxlen=maxlen, dtype='float32')
A2 = pad_sequences(A2, maxlen=maxlen, dtype='float32')
A3 = pad_sequences(A3, maxlen=maxlen, dtype='float32')
A4 = pad_sequences(A4, maxlen=maxlen, dtype='float32')
A5 = pad_sequences(A5, maxlen=maxlen, dtype='float32')


'''
prefix = './vec/'+args.split+'.'+args.type+'.'
passages = load_3d_array(prefix + 'passages')
questions = load_3d_array(prefix + 'questions')
A1 = load_3d_array(prefix + 'A1')
A2 = load_3d_array(prefix + 'A2')
A3 = load_3d_array(prefix + 'A3')
A4 = load_3d_array(prefix + 'A4')
A5 = load_3d_array(prefix + 'A5')
true_ans = load_2d_array(prefix + 'true_ans')
maxlen_pass = np.shape(passages)[1]
maxlen = np.shape(A1)[0]
'''


model = twoGRUmodel(args, maxlen, maxlen_pass, dim_glove)

total_data = len(A1)
step = args.batch_size*5/4
data_portion = total_data/step

print "Seperate to %d portions" % data_portion 
print "Training started..."
for e in range(args.epochs):
	print "Epoch %d" % e
#	index = np.arange(total_data)
#	np.random.shuffle(index)
	for start in range(0,total_data,step):
		end = start+step
		if end > total_data:
			end = total_data
		pdb.set_trace()
		model.fit(
			{'ques_input':questions[start:end], 'pass_input':passages[start:end], 'A1_input':A1[start:end], 'A2_input':A2[start:end], 'A3_input':A3[start:end], 'A4_input':A4[start:end], 'A5_input':A5[start:end]},
			{'final_out':true_ans[start:end]},
			batch_size=args.batch_size,
			nb_epoch=1,
			validation_split=0.2,
			#	validation_data=(
			#	{'ques_input':questions_val, 'pass_input':passages_val, 'A1_input':A1_val, 'A2_input':A2_val, 'A3_input':A3_val, 'A4_input':A4_val, 'A5_input':A5_val},
			#	{'final_out':true_ans_val})
			#shuffle=True,
		)
	
#pred_train = model.predict(
#	{'ques_input':questions, 'pass_input':passages, 'A1_input':A1, 'A2_input':A2, 'A3_input':A3, 'A4_input':A4, 'A5_input':A5},
#)
#pred_val = model.predict(
#	{'ques_input':questions_val, 'pass_input':passages_val, 'A1_input':A1_val, 'A2_input':A2_val, 'A3_input':A3_val, 'A4_input':A4_val, 'A5_input':A5_val},
#)

