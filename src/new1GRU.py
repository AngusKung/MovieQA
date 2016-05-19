import cPickle
import argparse
import numpy as np
import sys

"""my .py file"""
from parse import *
from myLayers import *

"""Keras"""
from keras.models import Sequential, Model 
from keras.layers.core import *
from keras.layers import *
from keras.regularizers import l2, activity_l2
from keras.layers.recurrent import LSTM, GRU
from keras.optimizers import SGD,RMSprop
from keras.utils import generic_utils
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences

iii=1
jjj=0
while jjj<iii:
	parser = argparse.ArgumentParser()
	parser.add_argument("-dim_gru", type=int, default=128)
	parser.add_argument("-n_hid_layers", type=int, default=2)
	parser.add_argument("-dropout", type=float, default=0)
	parser.add_argument("-activation", type=str, default="softplus")
	parser.add_argument("-epochs", type=int, default=1000)
	parser.add_argument("-model_save_interval", type=int, default=10)
	parser.add_argument("-batch_size", type=int, default=20)
	parser.add_argument("-lr", type=float, default=0.00001)
	parser.add_argument("-dataset", type=str, default="mod1")
	#parser.add_argument("-mode", type=str, required=True, help='Choose gru mode: sep or com', choices=['sep','com'])
	args = parser.parse_args()

	print "Running with args:"
	print args

	print "Loading MCTest training data...\n",
	training_data = cPickle.load(open("Pickle/train.lstm.pickle"))
	valid_data = cPickle.load(open("Pickle/val.lstm.pickle"))
	'''if(args.dataset==".x24"):
		training_data = cPickle.load(open("Pickle/mc500.train.lstm.x24.noStopWord.pickle"))
		valid_data = cPickle.load(open("Pickle/mc500.dev.lstm.noStopWord.pickle"))
	elif(args.dataset=="cos>0.75"):
		training_data = cPickle.load(open("Pickle/mc500+mc160.train.lstm.cos>0.75.pickle"))
		valid_data = cPickle.load(open("Pickle/mc500+mc160.dev.lstm.cos>0.75.pickle"))'''
	
	passages, questions, A1, A2, A3, A4, true_ans = mk_newgru300(training_data)
	passages_val, questions_val, A1_val, A2_val, A3_val, A4_val, true_ans_val = mk_newgru300(valid_data)
	#sys.exit(1)
	maxlen = 0
	# 3 stories more than 400 words in train(4xx,5xx,6xx), 2 stories more than 400 words in test 
	for ff in (A1):
		if len(ff) > maxlen:
			maxlen = len(ff)
	print "MAxlen A1"+str(maxlen)
	for ff in (questions):
		if len(ff) > maxlen:
			#print "train Line "+str(i)+" :"+str(len(ff))
			maxlen = len(ff)
	print "MAxlen Q"+str(maxlen)

	
	maxlen_pass = 0
	for i,ff in enumerate(passages):
		if len(ff) > maxlen_pass:
			#print "train Line "+str(i)+" :"+str(len(ff))
			maxlen_pass = len(ff)

	maxlen = maxlen_pass # for common GRU

	passages = pad_sequences(passages, maxlen=maxlen_pass, dtype='float32')
	passages_val = pad_sequences(passages_val, maxlen=maxlen_pass, dtype='float32')
	questions = pad_sequences(questions, maxlen=maxlen, dtype='float32')
	questions_val = pad_sequences(questions_val, maxlen=maxlen, dtype='float32')
	A1 = pad_sequences(A1, maxlen=maxlen, dtype='float32')
	A2 = pad_sequences(A2, maxlen=maxlen, dtype='float32')
	A3 = pad_sequences(A3, maxlen=maxlen, dtype='float32')
	A4 = pad_sequences(A4, maxlen=maxlen, dtype='float32')
	A1_val = pad_sequences(A1_val, maxlen=maxlen, dtype='float32')
	A2_val = pad_sequences(A2_val, maxlen=maxlen, dtype='float32')
	A3_val = pad_sequences(A3_val, maxlen=maxlen, dtype='float32')
	A4_val = pad_sequences(A4_val, maxlen=maxlen, dtype='float32')


	n_classes= 4 # number of classes to be classified == 4 choices
	dim_glove = 300
	dim_gru = args.dim_gru

	def mean_along_time(x):
		return K.mean(x,axis=1)



	shared_GRU =  GRU(output_dim = dim_gru, dropout_W=args.dropout, return_sequences = True, input_shape = (maxlen,dim_glove), init = 'glorot_uniform', inner_init = 'orthogonal', inner_activation = 'hard_sigmoid')
	shared_backGRU = GRU(output_dim = dim_gru,dropout_W=args.dropout ,go_backwards=True, return_sequences = True, input_shape = (maxlen,dim_glove), init = 'glorot_uniform', inner_init = 'orthogonal', inner_activation = 'hard_sigmoid')



	pass_input = Input(shape=(maxlen_pass,dim_glove), dtype='float32', name='pass_input')
	#pass_gru = GRU(output_dim = dim_gru, dropout_W=args.dropout, return_sequences = True, input_shape = (maxlen_pass,dim_glove), init = 'glorot_uniform', inner_init = 'orthogonal', inner_activation = 'hard_sigmoid')(pass_input) # maxlen_pass, dim_gru
	#pass_backgru = GRU(output_dim = dim_gru, dropout_W=args.dropout ,go_backwards=True, return_sequences = True, input_shape = (maxlen_pass,dim_glove), init = 'glorot_uniform', inner_init = 'orthogonal', inner_activation = 'hard_sigmoid')(pass_input) # maxlen_pass, dim_gru
	pass_gru = shared_GRU(pass_input)
	pass_backgru = shared_backGRU(pass_input)
	pass_con = merge([pass_gru,pass_backgru],mode='concat') # maxlen_pass, 2*dim_gru

	ques_input = Input(shape=(maxlen,dim_glove), dtype='float32', name='ques_input')
	ques_gru = shared_GRU(ques_input)
	ques_backgru = shared_backGRU(ques_input)
	ques_gru_last = Lambda(get_time_slice,get_time_slice_output_shape)(ques_gru)
	ques_backgru_last = Lambda(get_time_slice,get_time_slice_output_shape)(ques_backgru)
	ques_con = merge([ques_gru_last,ques_backgru_last],mode='concat') # , 2*dim_gru
	repeat_ques = RepeatVector(maxlen_pass)(ques_con)
	mul_ques_pass = merge([pass_con,repeat_ques],mode='mul') # maxlen_pass, 2*dim_gru
	permute_qp_mul = Permute((2,1))(mul_ques_pass) # 2*dim_gru, maxlen_pass
	#cos_ques_pass = Lambda(cos_sim_matvec,cos_sim_matvec_output_shape)([pass_con,ques_con]) # b_s, maxlen_pass
	dot_ques_pass = Lambda(sum_along_time,sum_along_time_output_shape)(permute_qp_mul) # maxlen_pass
	thecoef = Activation('softmax')(dot_ques_pass)
	repeat_coeff = RepeatVector(2*dim_gru)(thecoef) # 2*dim_gru, maxlen_pass
	permute_coeff = Permute((2,1))(repeat_coeff) # maxlen_pass, 2*dim_gru
	weighted_vec = merge([permute_coeff, pass_con],mode='mul') # maxlen_pass, 2*dim_gru
	atten_out = Lambda(sum_along_time,sum_along_time_output_shape)(weighted_vec) # 2*dim_gru

	
	#atten_coef = Activation('sigmoid')(cos_out) # maxlen_pass
	#atten_coef_mat = RepeatVector(dim_glove)(atten_coef) #dim_glove, maxlen_pass
	#atten_coef_mat_trans = Permute((2,1))(atten_coef_mat) # maxlen_pass, dim_glove
	#atten_mul = merge([atten_coef_mat_trans, pass_input], mode='mul') # maxlen_pass, dim_glove
	#atten_out = Lambda(sum_along_time,sum_along_time_output_shape)(atten_mul) # dim_glove

	A1_input = Input(shape=(maxlen,dim_glove),name='A1_input',dtype='float32') # dim_glove
	A2_input = Input(shape=(maxlen,dim_glove),name='A2_input',dtype='float32') # dim_glove
	A3_input = Input(shape=(maxlen,dim_glove),name='A3_input',dtype='float32') # dim_glove
	A4_input = Input(shape=(maxlen,dim_glove),name='A4_input',dtype='float32') # dim_glove

	
	a1_gru = shared_GRU(A1_input)
	a1_backgru = shared_backGRU(A1_input)
	a1_gru_last = Lambda(get_time_slice,get_time_slice_output_shape)(a1_gru)
	a1_backgru_last = Lambda(get_time_slice,get_time_slice_output_shape)(a1_backgru)
	a1_con = merge([a1_gru_last,a1_backgru_last],mode='concat') # , 2*dim_gru

	a2_gru = shared_GRU(A2_input)
	a2_backgru = shared_backGRU(A2_input)
	a2_gru_last = Lambda(get_time_slice,get_time_slice_output_shape)(a2_gru)
	a2_backgru_last = Lambda(get_time_slice,get_time_slice_output_shape)(a2_backgru)
	a2_con = merge([a2_gru_last,a2_backgru_last],mode='concat') # , 2*dim_gru

	a3_gru = shared_GRU(A3_input)
	a3_backgru = shared_backGRU(A3_input)
	a3_gru_last = Lambda(get_time_slice,get_time_slice_output_shape)(a3_gru)
	a3_backgru_last = Lambda(get_time_slice,get_time_slice_output_shape)(a3_backgru)
	a3_con = merge([a3_gru_last,a3_backgru_last],mode='concat') # , 2*dim_gru

	a4_gru = shared_GRU(A4_input)
	a4_backgru = shared_backGRU(A4_input)
	a4_gru_last = Lambda(get_time_slice,get_time_slice_output_shape)(a4_gru)
	a4_backgru_last = Lambda(get_time_slice,get_time_slice_output_shape)(a4_backgru)
	a4_con = merge([a4_gru_last,a4_backgru_last],mode='concat') # , 2*dim_gru


	#A1_mul = merge([a1_con,atten_out], mode='cos', dot_axes=[1,1])  # (batch_size, dim_fuck)
	#A1_mul = Lambda(cos_sim,output_shape=cos_sim_output_shape)([a1_con,atten_out])  # (batch_size, dim_fuck)
	#A2_mul = Lambda(cos_sim,output_shape=cos_sim_output_shape)([a2_con,atten_out])  # (batch_size, dim_fuck)
	#A3_mul = Lambda(cos_sim,output_shape=cos_sim_output_shape)([a3_con,atten_out])  # (batch_size, dim_fuck)
	#A4_mul = Lambda(cos_sim,output_shape=cos_sim_output_shape)([a4_con,atten_out])  # (batch_size, dim_fuck)

	A1_mul = merge([a1_con,atten_out], mode='mul')  # (batch_size, dim_fuck)
	A2_mul = merge([a2_con,atten_out], mode='mul')
	A3_mul = merge([a3_con,atten_out], mode='mul')
	A4_mul = merge([a4_con,atten_out], mode='mul')
	
	A1_out = Lambda(sum_one,sum_one_output_shape)(A1_mul)
	A2_out = Lambda(sum_one,sum_one_output_shape)(A2_mul)
	A3_out = Lambda(sum_one,sum_one_output_shape)(A3_mul)
	A4_out = Lambda(sum_one,sum_one_output_shape)(A4_mul)

	merge_out = merge([A1_out,A2_out,A3_out,A4_out],mode='concat')
	#final_out = merge([A1_mul,A2_mul,A3_mul,A4_mul],mode='concat',name='final_out')
	

	#final_out = Dense(4,activation='softmax',name='final_out')(concon)
	final_out = Activation('softmax',name='final_out')(merge_out)
	model = Model(input=[ques_input,pass_input,A1_input,A2_input,A3_input,A4_input],output=[final_out])
	#model = Model(input=[ques_input,pass_input],output=[final_out])
	
	print "Compiling model..."
        rmsprop = RMSprop(lr=args.lr)
        #rmsprop = RMSprop(lr=0.0001)
	model.compile(loss={'final_out':"categorical_crossentropy"}, optimizer=rmsprop,metrics=['accuracy'])
	print "Compilation done..."
	
	print "Training started..."
	
	model.fit(
		{'ques_input':questions, 'pass_input':passages, 'A1_input':A1, 'A2_input':A2, 'A3_input':A3, 'A4_input':A4},
		{'final_out':true_ans},
		batch_size=args.batch_size,
		nb_epoch=args.epochs,
		#validation_split=0.1,
		validation_data=(
		{'ques_input':questions_val, 'pass_input':passages_val, 'A1_input':A1_val, 'A2_input':A2_val, 'A3_input':A3_val, 'A4_input':A4_val},
		{'final_out':true_ans_val})
		#shuffle=True,
	)
	
	fuck_train = model.predict(
		{'ques_input':questions, 'pass_input':passages, 'A1_input':A1, 'A2_input':A2, 'A3_input':A3, 'A4_input':A4},
	)
	fuck_val = model.predict(
		{'ques_input':questions_val, 'pass_input':passages_val, 'A1_input':A1_val, 'A2_input':A2_val, 'A3_input':A3_val, 'A4_input':A4_val},
	)
	jjj =jjj+1


