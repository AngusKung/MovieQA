import argparse
import numpy as np
import sys
import cPickle

"""my .py file"""
from parse import mk_newgru300, save_3d_array, load_3d_array, save_2d_array, load_2d_array
from mybiGRU import loadMQAPickle,findMaxlen

"""Keras"""
from keras.preprocessing.sequence import pad_sequences

parser = argparse.ArgumentParser()
parser.add_argument("-path",type=str,required=True)
parser.add_argument("-split",type=str, choices=['train','val','test'],required=True)
parser.add_argument("-type", type=str, choices=['plot','dvs','subtitle','script'] , required=True)
args = parser.parse_args()

path = args.path

print "Calling preprocessing on "+path

print "Loading MovieQA data..."
#training_data, valid_data = loadMQAPickle(args)
training_data  = loadMQAPickle(path)
passages, questions, A1, A2, A3, A4, A5, true_ans = mk_newgru300(training_data)
#passages, questions, A1, A2, A3, A4, A5, true_ans = mk_newgru300(training_data)
#passages_val, questions_val, A1_val, A2_val, A3_val, A4_val, A5_val, true_ans_val = mk_newgru300(valid_data)

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

prefix = './vec/'+args.split+'.'+args.type+'.'

save_3d_array(prefix + 'passages',passages)
save_3d_array(prefix + 'questions',questions)
save_3d_array(prefix + 'A1',A1)
save_3d_array(prefix + 'A2',A2)
save_3d_array(prefix + 'A3',A3)
save_3d_array(prefix + 'A4',A4)
save_3d_array(prefix + 'A5',A5)
save_2d_array(prefix + 'true_ans', true_ans)


#prefix = './vec/'+args.split+'.'+args.type+'.'
#save_2d_array(prefix + 'true_ans', true_ans)

#newtrue = load_2d_array(prefix + 'true_ans')

#new_A1 = load_3d_array(prefix + 'A1')
#print np.all(newtrue == true_ans)
#print np.shape(new_A1)
