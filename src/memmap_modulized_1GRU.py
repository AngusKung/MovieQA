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
parser.add_argument("-epochs", type=int, default=500)
parser.add_argument("-model_save_interval", type=int, default=10)
parser.add_argument("-batch_size", type=int, default=64)
parser.add_argument("-lr", type=float, default=0.001)
parser.add_argument("-hop", type=int, default=0)
parser.add_argument("-atten_mode", type=str, choices=['sigmoid','softmax'], default="softmax")
parser.add_argument("-path",type=str,required=True)
parser.add_argument("-split",type=str, choices=['train','val','test'],required=True)
parser.add_argument("-type", type=str, choices=['plot','dvs','subtitle','script'] , required=True)
args = parser.parse_args()

print "Running with args:"
print args

dim_glove = 100
#path = './Pickle/val.plot.lstm.pickle'


print "Loading MovieQA data..."

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
Qnum = 9848
maxlen = 32
maxlen_pass = 265

path_name = args.path
passMemmap_name = path_name+"pass.memmap"
queMemmap_name = path_name+"que.memmap"
A1Memmap_name = path_name+"A1.memmap"
A2Memmap_name = path_name+"A2.memmap"
A3Memmap_name = path_name+"A3.memmap"
A4Memmap_name = path_name+"A4.memmap"
A5Memmap_name = path_name+"A5.memmap"
ansMemmap_name = path_name+"ans.memmap"

passages = np.memmap(passMemmap_name, dtype='float32', mode='r+', shape=(Qnum,maxlen_pass,dim_glove))
questions = np.memmap(queMemmap_name, dtype='float32', mode='r+', shape=(Qnum,maxlen,dim_glove))
A1 = np.memmap(A1Memmap_name, dtype='float32', mode='r+', shape=(Qnum,maxlen,dim_glove))
A2 = np.memmap(A2Memmap_name, dtype='float32', mode='r+', shape=(Qnum,maxlen,dim_glove))
A3 = np.memmap(A3Memmap_name, dtype='float32', mode='r+', shape=(Qnum,maxlen,dim_glove))
A4 = np.memmap(A4Memmap_name, dtype='float32', mode='r+', shape=(Qnum,maxlen,dim_glove))
A5 = np.memmap(A5Memmap_name, dtype='float32', mode='r+', shape=(Qnum,maxlen,dim_glove))
true_ans = np.memmap(ansMemmap_name, dtype='float32', mode='r+', shape=(Qnum,5))

val_Qnum = 1958
val_maxlen = 32
val_maxlen_pass = 265

val_path_name = "./Memmap/val.plot.5.100d.mp=265.m=32.Q=1958.lstm/"
val_passMemmap_name = val_path_name+"pass.memmap"
val_queMemmap_name = val_path_name+"que.memmap"
val_A1Memmap_name = val_path_name+"A1.memmap"
val_A2Memmap_name = val_path_name+"A2.memmap"
val_A3Memmap_name = val_path_name+"A3.memmap"
val_A4Memmap_name = val_path_name+"A4.memmap"
val_A5Memmap_name = val_path_name+"A5.memmap"
val_ansMemmap_name = val_path_name+"ans.memmap"

val_passages = np.memmap(val_passMemmap_name, dtype='float32', mode='r+', shape=(val_Qnum,val_maxlen_pass,dim_glove))
val_questions = np.memmap(val_queMemmap_name, dtype='float32', mode='r+', shape=(val_Qnum,val_maxlen,dim_glove))
val_A1 = np.memmap(val_A1Memmap_name, dtype='float32', mode='r+', shape=(val_Qnum,val_maxlen,dim_glove))
val_A2 = np.memmap(val_A2Memmap_name, dtype='float32', mode='r+', shape=(val_Qnum,val_maxlen,dim_glove))
val_A3 = np.memmap(val_A3Memmap_name, dtype='float32', mode='r+', shape=(val_Qnum,val_maxlen,dim_glove))
val_A4 = np.memmap(val_A4Memmap_name, dtype='float32', mode='r+', shape=(val_Qnum,val_maxlen,dim_glove))
val_A5 = np.memmap(val_A5Memmap_name, dtype='float32', mode='r+', shape=(val_Qnum,val_maxlen,dim_glove))
val_true_ans = np.memmap(val_ansMemmap_name, dtype='float32', mode='r+', shape=(val_Qnum,5))

model = twoGRUmodel(args, maxlen, maxlen_pass, dim_glove)
#pdb.set_trace()
print "Training started..."
model.fit(
	{'ques_input':questions, 'pass_input':passages, 'A1_input':A1, 'A2_input':A2, 'A3_input':A3, 'A4_input':A4, 'A5_input':A5},
	{'final_out':true_ans},
	batch_size=args.batch_size,
	nb_epoch = args.epochs,
	#validation_split=0.2,
				validation_data=(
				{'ques_input':val_questions, 'pass_input':val_passages, 'A1_input':val_A1, 'A2_input':val_A2, 'A3_input':val_A3, 'A4_input':val_A4, 'A5_input':val_A5},
				{'final_out':val_true_ans}),
			shuffle=True
	)
	
#pred_train = model.predict(
#	{'ques_input':questions, 'pass_input':passages, 'A1_input':A1, 'A2_input':A2, 'A3_input':A3, 'A4_input':A4, 'A5_input':A5},
#)
#pred_val = model.predict(
#	{'ques_input':questions_val, 'pass_input':passages_val, 'A1_input':A1_val, 'A2_input':A2_val, 'A3_input':A3_val, 'A4_input':A4_val, 'A5_input':A5_val},
#)

