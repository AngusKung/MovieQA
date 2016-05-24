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
parser.add_argument("-path_data",type=str,required=True)
parser.add_argument("-path_ans",type=str,required=True)
parser.add_argument("-split",type=str, choices=['train','val','test'],required=True)
parser.add_argument("-type", type=str, choices=['plot','dvs','subtitle','script'] , required=True)
args = parser.parse_args()

print "Running with args:"
print args

dim_glove = 300
#path = './Pickle/val.plot.lstm.pickle'


print "Loading MovieQA data by memmap..."
dataM = np.memmap(args.path_data, dtype='float32', mode='r', shape=(8577998,300))

ansM = np.memmap(args.path_ans, dtype='float32', mode='r', shape=(1958,5))

maxlen_pass = 3643
maxlen = 123
aQ = maxlen_pass + maxlen*6
Qnum = 1958

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
pdb.set_trace()


print "Training started..."
for e in range(args.epochs):
	print "Epoch %d" % e
	index = np.arange(Qnum)
	np.random.shuffle(index)
        #pickNum = index[0]
	train_loss = 0
        for idx in index:
        	start = (idx) *aQ
        	end = (idx+1) * aQ
		train_loss += model.train_on_batch(
			{'ques_input':dataM[start+maxlen_pass:start+maxlen_pass+maxlen], 'pass_input':dataM[start:start+maxlen_pass], 'A1_input':dataM[start+maxlen_pass+maxlen:start+maxlen_pass+2*maxlen], 'A2_input':dataM[start+maxlen_pass+2*maxlen:start+maxlen_pass+3*maxlen], 'A3_input':dataM[start+maxlen_pass+3*maxlen:start+maxlen_pass+4*maxlen], 'A4_input':dataM[start+maxlen_pass+4*maxlen:start+maxlen_pass+5*maxlen], 'A5_input':dataM[start+maxlen_pass+5*maxlen:start+maxlen_pass+6*maxlen]},
			{'final_out':ansM[idx]},
			#	validation_data=(
			#	{'ques_input':questions_val, 'pass_input':passages_val, 'A1_input':A1_val, 'A2_input':A2_val, 'A3_input':A3_val, 'A4_input':A4_val, 'A5_input':A5_val},
			#	{'final_out':true_ans_val})
			#shuffle=True,
		)
		print "Now on",i+1,"   loss = ",train_loss/(i+1)
		sys.stdout.write("\033[F")
	print "training loss = ",train_loss/Qnum
		#model.predict
	
#pred_train = model.predict(
#	{'ques_input':questions, 'pass_input':passages, 'A1_input':A1, 'A2_input':A2, 'A3_input':A3, 'A4_input':A4, 'A5_input':A5},
#)
#pred_val = model.predict(
#	{'ques_input':questions_val, 'pass_input':passages_val, 'A1_input':A1_val, 'A2_input':A2_val, 'A3_input':A3_val, 'A4_input':A4_val, 'A5_input':A5_val},
#)

