import argparse
import numpy as np
import sys
import cPickle

"""my .py file"""
from parse import mk_newgru300
from mybiGRU import oneGRUmodel, twoGRUmodel,loadMQAPickle,findMaxlen,MQA_datagen


"""Keras"""
from keras.preprocessing.sequence import pad_sequences
from keras.engine.training import slice_X

parser = argparse.ArgumentParser()
parser.add_argument("-dim_gru", type=int, default=128)
parser.add_argument("-n_hid_layers", type=int, default=2)
parser.add_argument("-dropout", type=float, default=0)
parser.add_argument("-activation", type=str, default="softplus")
parser.add_argument("-epochs", type=int, default=1000)
parser.add_argument("-model_save_interval", type=int, default=10)
parser.add_argument("-batch_size", type=int, default=20)
parser.add_argument("-lr", type=float, default=0.001)
parser.add_argument("-dataset", type=str, default="mod1")
parser.add_argument("-hop", type=int, default=0)
parser.add_argument("-atten_mode", type=str, choices=['sigmoid','softmax'], default="sigmoid")
args = parser.parse_args()

print "Running with args:"
print args

dim_glove = 300
path = './Pickle/val.plot.lstm.pickle'

'''
print "Loading MovieQA data..."
#training_data, valid_data = loadMQAPickle(args)
training_data  = loadMQAPickle(args)
passages, questions, A1, A2, A3, A4, A5, true_ans = mk_newgru300(training_data)
#passages, questions, A1, A2, A3, A4, A5, true_ans = mk_newgru300(training_data)
#passages_val, questions_val, A1_val, A2_val, A3_val, A4_val, true_ans_val = mk_newgru300(valid_data)

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
#passages_val = pad_sequences(passages_val, maxlen=maxlen_pass, dtype='float32')
questions = pad_sequences(questions, maxlen=maxlen, dtype='float32')
#questions_val = pad_sequences(questions_val, maxlen=maxlen, dtype='float32')
A1 = pad_sequences(A1, maxlen=maxlen, dtype='float32')
A2 = pad_sequences(A2, maxlen=maxlen, dtype='float32')
A3 = pad_sequences(A3, maxlen=maxlen, dtype='float32')
A4 = pad_sequences(A4, maxlen=maxlen, dtype='float32')
A5 = pad_sequences(A5, maxlen=maxlen, dtype='float32')

#A1_val = pad_sequences(A1_val, maxlen=maxlen, dtype='float32')
#A2_val = pad_sequences(A2_val, maxlen=maxlen, dtype='float32')
#A3_val = pad_sequences(A3_val, maxlen=maxlen, dtype='float32')
#A4_val = pad_sequences(A4_val, maxlen=maxlen, dtype='float32')
'''

f = open(path, "rb")
num_data = cPickle.load(f)
maxlen_pass = cPickle.load(f)
maxlen_ques = cPickle.load(f)
maxlen = cPickle.load(f)
f.close()

model = twoGRUmodel(args, max(maxlen,maxlen_ques), maxlen_pass, dim_glove)
#sys.exit(-1)

print "Training started..."

validation_split = 0.2
split_at = int(num_data*(1.-validation_split))
train_ind = np.arange(split_at)
val_ind = np.arange(split_at, num_data)

model.fit_generator(
	MQA_datagen(path, args, 0, split_at),
	samples_per_epoch = len(train_ind),
	nb_epoch=args.epochs,
	validation_data=MQA_datagen(path, args, split_at, num_data),
	nb_val_samples = len(val_ind),
	max_q_size =20
#	validation_data=(
#	{'ques_input':questions_val, 'pass_input':passages_val, 'A1_input':A1_val, 'A2_input':A2_val, 'A3_input':A3_val, 'A4_input':A4_val},
#	{'final_out':true_ans_val})
)
	
#pred_train = model.predict(
#	{'ques_input':questions, 'pass_input':passages, 'A1_input':A1, 'A2_input':A2, 'A3_input':A3, 'A4_input':A4},
#)
#pred_val = model.predict(
#	{'ques_input':questions_val, 'pass_input':passages_val, 'A1_input':A1_val, 'A2_input':A2_val, 'A3_input':A3_val, 'A4_input':A4_val},
#)

