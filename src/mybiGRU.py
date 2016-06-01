import numpy as np
import cPickle
import random

"""my .py file"""
from myLayers import *
from parse import mk_newgru300

"""Keras"""
from keras.models import Model 
from keras.layers.core import *
from keras.layers import *
from keras.optimizers import SGD,RMSprop
from keras.layers.recurrent import LSTM, GRU
from keras.utils import generic_utils
from keras.preprocessing.sequence import pad_sequences

def findMaxlen(x_list,init_max=0):
	maxlen = init_max
	for x in x_list:
		if len(x) > maxlen:
			maxlen = len(x)
	return maxlen


def MQA_datagen(path,args,sta,end):
	f = open(path, "rb")
	num_data = cPickle.load(f)
	maxlen_pass = cPickle.load(f)
	maxlen_ques = cPickle.load(f)
	maxlen = cPickle.load(f)
	data =  cPickle.load(f)
	maxlen = max(maxlen,maxlen_ques)
	# for split
	data = data[sta:end]	
	# for data shuffle
	random.shuffle(data)
	batch_index = 0
	while True:
		#batch_index = np.random.randint(0, len(data)-args.batch_size )
		start = batch_index
		end = start + args.batch_size
		
		passages, questions, A1, A2, A3, A4, A5, true_ans = mk_newgru300(data[start:end])
		passages = pad_sequences(passages, maxlen=maxlen_pass, dtype='float32')
		questions = pad_sequences(questions, maxlen=maxlen, dtype='float32')
		A1 = pad_sequences(A1, maxlen=maxlen, dtype='float32')
		A2 = pad_sequences(A2, maxlen=maxlen, dtype='float32')
		A3 = pad_sequences(A3, maxlen=maxlen, dtype='float32')
		A4 = pad_sequences(A4, maxlen=maxlen, dtype='float32')
		A5 = pad_sequences(A5, maxlen=maxlen, dtype='float32')
		
		batch_index = batch_index + args.batch_size

		yield ({'ques_input':questions, 'pass_input':passages, 'A1_input':A1, 'A2_input':A2, 'A3_input':A3, 'A4_input':A4, 'A5_input':A5},
		{'final_out':true_ans}
		)
		
	f.close()
	#return valid_data

def twoLSTMmodel(args, maxlen, maxlen_pass, dim_glove):
	#maxlen_pass = maxlen
	dim_gru = args.dim_gru
	#dim_glove = 300
	shared_GRU =  LSTM(output_dim = dim_gru, dropout_W=args.dropout, return_sequences = False, input_shape = (maxlen,dim_glove), init = 'glorot_uniform', inner_init = 'orthogonal', inner_activation = 'hard_sigmoid')
	shared_backGRU = LSTM(output_dim = dim_gru,dropout_W=args.dropout ,go_backwards=True, return_sequences = False, input_shape = (maxlen,dim_glove), init = 'glorot_uniform', inner_init = 'orthogonal', inner_activation = 'hard_sigmoid')
	passGRU =  LSTM(output_dim = dim_gru, dropout_W=args.dropout, return_sequences = True, input_shape = (maxlen_pass,dim_glove), init = 'glorot_uniform', inner_init = 'orthogonal', inner_activation = 'hard_sigmoid')
	passbackGRU = LSTM(output_dim = dim_gru,dropout_W=args.dropout ,go_backwards=True, return_sequences = True, input_shape = (maxlen_pass,dim_glove), init = 'glorot_uniform', inner_init = 'orthogonal', inner_activation = 'hard_sigmoid')


	pass_input = Input(shape=(maxlen_pass,dim_glove), dtype='float32', name='pass_input')
	pass_gru = passGRU(pass_input)
	pass_backgru = passbackGRU(pass_input)
	pass_con = merge([pass_gru,pass_backgru],mode='concat') # maxlen_pass, 2*dim_gru

	ques_input = Input(shape=(maxlen,dim_glove), dtype='float32', name='ques_input')
	ques_gru = shared_GRU(ques_input)
	ques_backgru = shared_backGRU(ques_input)
	ques_con = merge([ques_gru,ques_backgru],mode='concat') # , 2*dim_gru

	repeat_ques = RepeatVector(maxlen_pass)(ques_con)
	mul_ques_pass = merge([pass_con,repeat_ques],mode='mul') # maxlen_pass, 2*dim_gru
	permute_qp_mul = Permute((2,1))(mul_ques_pass) # 2*dim_gru, maxlen_pass
	#cos_ques_pass = Lambda(cos_sim_matvec,cos_sim_matvec_output_shape)([pass_con,ques_con]) # b_s, maxlen_pass
	dot_ques_pass = Lambda(sum_along_time,sum_along_time_output_shape)(permute_qp_mul) # maxlen_pass
	thecoef = Activation(args.atten_mode)(dot_ques_pass)
	repeat_coeff = RepeatVector(2*dim_gru)(thecoef) # 2*dim_gru, maxlen_pass
	permute_coeff = Permute((2,1))(repeat_coeff) # maxlen_pass, 2*dim_gru
	weighted_vec = merge([permute_coeff, pass_con],mode='mul') # maxlen_pass, 2*dim_gru
	atten_out = Lambda(sum_along_time,sum_along_time_output_shape)(weighted_vec) # 2*dim_gru

	A1_input = Input(shape=(maxlen,dim_glove),name='A1_input',dtype='float32') # dim_glove
	A2_input = Input(shape=(maxlen,dim_glove),name='A2_input',dtype='float32') # dim_glove
	A3_input = Input(shape=(maxlen,dim_glove),name='A3_input',dtype='float32') # dim_glove
	A4_input = Input(shape=(maxlen,dim_glove),name='A4_input',dtype='float32') # dim_glove
	A5_input = Input(shape=(maxlen,dim_glove),name='A5_input',dtype='float32') # dim_glove

	
	a1_gru = shared_GRU(A1_input)
	a1_backgru = shared_backGRU(A1_input)
	a1_con = merge([a1_gru,a1_backgru],mode='concat') # , 2*dim_gru

	a2_gru = shared_GRU(A2_input)
	a2_backgru = shared_backGRU(A2_input)
	a2_con = merge([a2_gru,a2_backgru],mode='concat') # , 2*dim_gru

	a3_gru = shared_GRU(A3_input)
	a3_backgru = shared_backGRU(A3_input)
	a3_con = merge([a3_gru,a3_backgru],mode='concat') # , 2*dim_gru

	a4_gru = shared_GRU(A4_input)
	a4_backgru = shared_backGRU(A4_input)
	a4_con = merge([a4_gru,a4_backgru],mode='concat') # , 2*dim_gru

	a5_gru = shared_GRU(A5_input)
	a5_backgru = shared_backGRU(A5_input)
	a5_con = merge([a5_gru,a5_backgru],mode='concat') # , 2*dim_gru
#	a1_con = Lambda(the_l2, the_l2_output_shape)(a1_con)
#	a2_con = Lambda(the_l2, the_l2_output_shape)(a2_con)
#	a3_con = Lambda(the_l2, the_l2_output_shape)(a3_con)
#	a4_con = Lambda(the_l2, the_l2_output_shape)(a4_con)


	#A1_mul = Lambda(cos_sim,output_shape=cos_sim_output_shape)([a1_con,atten_out])  # (batch_size, dim_fuck)
	#A2_mul = Lambda(cos_sim,output_shape=cos_sim_output_shape)([a2_con,atten_out])  # (batch_size, dim_fuck)
	#A3_mul = Lambda(cos_sim,output_shape=cos_sim_output_shape)([a3_con,atten_out])  # (batch_size, dim_fuck)
	#A4_mul = Lambda(cos_sim,output_shape=cos_sim_output_shape)([a4_con,atten_out])  # (batch_size, dim_fuck)

	for i in range(args.hop):
		add = merge([atten_out,ques_con],mode='sum')
		ques_con = Lambda(lambda x: x/2)(add)
#		ques_con = Lambda(the_l2, the_l2_output_shape)(ques_con)
		repeat_ques = RepeatVector(maxlen_pass)(ques_con)
		mul_ques_pass = merge([pass_con,repeat_ques],mode='mul') # maxlen_pass, 2*dim_gru
		permute_qp_mul = Permute((2,1))(mul_ques_pass) # 2*dim_gru, maxlen_pass
		#cos_ques_pass = Lambda(cos_sim_matvec,cos_sim_matvec_output_shape)([pass_con,ques_con]) # b_s, maxlen_pass
		dot_ques_pass = Lambda(sum_along_time,sum_along_time_output_shape)(permute_qp_mul) # maxlen_pass
		thecoef = Activation(args.atten_mode)(dot_ques_pass)
		repeat_coeff = RepeatVector(2*dim_gru)(thecoef) # 2*dim_gru, maxlen_pass
		permute_coeff = Permute((2,1))(repeat_coeff) # maxlen_pass, 2*dim_gru
		weighted_vec = merge([permute_coeff, pass_con],mode='mul') # maxlen_pass, 2*dim_gru
		atten_out = Lambda(sum_along_time,sum_along_time_output_shape)(weighted_vec) # 2*dim_gru
#		atten_out = Lambda(the_l2,the_l2_output_shape)(atten_out)

	A1_mul = merge([a1_con,atten_out], mode='mul')  # (batch_size, dim_fuck)
	A2_mul = merge([a2_con,atten_out], mode='mul')
	A3_mul = merge([a3_con,atten_out], mode='mul')
	A4_mul = merge([a4_con,atten_out], mode='mul')
	A5_mul = merge([a5_con,atten_out], mode='mul')
	
	A1_out = Lambda(sum_one,sum_one_output_shape)(A1_mul)
	A2_out = Lambda(sum_one,sum_one_output_shape)(A2_mul)
	A3_out = Lambda(sum_one,sum_one_output_shape)(A3_mul)
	A4_out = Lambda(sum_one,sum_one_output_shape)(A4_mul)
	A5_out = Lambda(sum_one,sum_one_output_shape)(A5_mul)

	merge_out = merge([A1_out,A2_out,A3_out,A4_out,A5_out],mode='concat')
	
	final_out = Activation('softmax',name='final_out')(merge_out)
	model = Model(input=[ques_input,pass_input,A1_input,A2_input,A3_input,A4_input,A5_input],output=[final_out])
	
	print "Compiling model..."
        rmsprop = RMSprop(lr=args.lr)
	model.compile(loss={'final_out':"categorical_crossentropy"}, optimizer=rmsprop,metrics=['accuracy'])
	print "Compilation done..."
	
	return model

def twoGRUmodel(args, maxlen, maxlen_pass, dim_glove):
	#maxlen_pass = maxlen
	dim_gru = args.dim_gru
	#dim_glove = 300
	shared_GRU =  GRU(output_dim = dim_gru, dropout_W=args.dropout, return_sequences = False, input_shape = (maxlen,dim_glove), init = 'glorot_uniform', inner_init = 'orthogonal', inner_activation = 'hard_sigmoid')
	shared_backGRU = GRU(output_dim = dim_gru,dropout_W=args.dropout ,go_backwards=True, return_sequences = False, input_shape = (maxlen,dim_glove), init = 'glorot_uniform', inner_init = 'orthogonal', inner_activation = 'hard_sigmoid')
	passGRU =  GRU(output_dim = dim_gru, dropout_W=args.dropout, return_sequences = True, input_shape = (maxlen_pass,dim_glove), init = 'glorot_uniform', inner_init = 'orthogonal', inner_activation = 'hard_sigmoid')
	passbackGRU = GRU(output_dim = dim_gru,dropout_W=args.dropout ,go_backwards=True, return_sequences = True, input_shape = (maxlen_pass,dim_glove), init = 'glorot_uniform', inner_init = 'orthogonal', inner_activation = 'hard_sigmoid')


	pass_input = Input(shape=(maxlen_pass,dim_glove), dtype='float32', name='pass_input')
	pass_gru = passGRU(pass_input)
	pass_backgru = passbackGRU(pass_input)
	pass_con = merge([pass_gru,pass_backgru],mode='concat') # maxlen_pass, 2*dim_gru

	ques_input = Input(shape=(maxlen,dim_glove), dtype='float32', name='ques_input')
	ques_gru = shared_GRU(ques_input)
	ques_backgru = shared_backGRU(ques_input)
	ques_con = merge([ques_gru,ques_backgru],mode='concat') # , 2*dim_gru

	repeat_ques = RepeatVector(maxlen_pass)(ques_con)
	mul_ques_pass = merge([pass_con,repeat_ques],mode='mul') # maxlen_pass, 2*dim_gru
	permute_qp_mul = Permute((2,1))(mul_ques_pass) # 2*dim_gru, maxlen_pass
	#cos_ques_pass = Lambda(cos_sim_matvec,cos_sim_matvec_output_shape)([pass_con,ques_con]) # b_s, maxlen_pass
	dot_ques_pass = Lambda(sum_along_time,sum_along_time_output_shape)(permute_qp_mul) # maxlen_pass
	thecoef = Activation(args.atten_mode)(dot_ques_pass)
	repeat_coeff = RepeatVector(2*dim_gru)(thecoef) # 2*dim_gru, maxlen_pass
	permute_coeff = Permute((2,1))(repeat_coeff) # maxlen_pass, 2*dim_gru
	weighted_vec = merge([permute_coeff, pass_con],mode='mul') # maxlen_pass, 2*dim_gru
	atten_out = Lambda(sum_along_time,sum_along_time_output_shape)(weighted_vec) # 2*dim_gru

	A1_input = Input(shape=(maxlen,dim_glove),name='A1_input',dtype='float32') # dim_glove
	A2_input = Input(shape=(maxlen,dim_glove),name='A2_input',dtype='float32') # dim_glove
	A3_input = Input(shape=(maxlen,dim_glove),name='A3_input',dtype='float32') # dim_glove
	A4_input = Input(shape=(maxlen,dim_glove),name='A4_input',dtype='float32') # dim_glove
	A5_input = Input(shape=(maxlen,dim_glove),name='A5_input',dtype='float32') # dim_glove

	
	a1_gru = shared_GRU(A1_input)
	a1_backgru = shared_backGRU(A1_input)
	a1_con = merge([a1_gru,a1_backgru],mode='concat') # , 2*dim_gru

	a2_gru = shared_GRU(A2_input)
	a2_backgru = shared_backGRU(A2_input)
	a2_con = merge([a2_gru,a2_backgru],mode='concat') # , 2*dim_gru

	a3_gru = shared_GRU(A3_input)
	a3_backgru = shared_backGRU(A3_input)
	a3_con = merge([a3_gru,a3_backgru],mode='concat') # , 2*dim_gru

	a4_gru = shared_GRU(A4_input)
	a4_backgru = shared_backGRU(A4_input)
	a4_con = merge([a4_gru,a4_backgru],mode='concat') # , 2*dim_gru

	a5_gru = shared_GRU(A5_input)
	a5_backgru = shared_backGRU(A5_input)
	a5_con = merge([a5_gru,a5_backgru],mode='concat') # , 2*dim_gru
#	a1_con = Lambda(the_l2, the_l2_output_shape)(a1_con)
#	a2_con = Lambda(the_l2, the_l2_output_shape)(a2_con)
#	a3_con = Lambda(the_l2, the_l2_output_shape)(a3_con)
#	a4_con = Lambda(the_l2, the_l2_output_shape)(a4_con)


	#A1_mul = Lambda(cos_sim,output_shape=cos_sim_output_shape)([a1_con,atten_out])  # (batch_size, dim_fuck)
	#A2_mul = Lambda(cos_sim,output_shape=cos_sim_output_shape)([a2_con,atten_out])  # (batch_size, dim_fuck)
	#A3_mul = Lambda(cos_sim,output_shape=cos_sim_output_shape)([a3_con,atten_out])  # (batch_size, dim_fuck)
	#A4_mul = Lambda(cos_sim,output_shape=cos_sim_output_shape)([a4_con,atten_out])  # (batch_size, dim_fuck)

	for i in range(args.hop):
		add = merge([atten_out,ques_con],mode='sum')
		ques_con = Lambda(lambda x: x/2)(add)
#		ques_con = Lambda(the_l2, the_l2_output_shape)(ques_con)
		repeat_ques = RepeatVector(maxlen_pass)(ques_con)
		mul_ques_pass = merge([pass_con,repeat_ques],mode='mul') # maxlen_pass, 2*dim_gru
		permute_qp_mul = Permute((2,1))(mul_ques_pass) # 2*dim_gru, maxlen_pass
		#cos_ques_pass = Lambda(cos_sim_matvec,cos_sim_matvec_output_shape)([pass_con,ques_con]) # b_s, maxlen_pass
		dot_ques_pass = Lambda(sum_along_time,sum_along_time_output_shape)(permute_qp_mul) # maxlen_pass
		thecoef = Activation(args.atten_mode)(dot_ques_pass)
		repeat_coeff = RepeatVector(2*dim_gru)(thecoef) # 2*dim_gru, maxlen_pass
		permute_coeff = Permute((2,1))(repeat_coeff) # maxlen_pass, 2*dim_gru
		weighted_vec = merge([permute_coeff, pass_con],mode='mul') # maxlen_pass, 2*dim_gru
		atten_out = Lambda(sum_along_time,sum_along_time_output_shape)(weighted_vec) # 2*dim_gru
#		atten_out = Lambda(the_l2,the_l2_output_shape)(atten_out)

	A1_mul = merge([a1_con,atten_out], mode='mul')  # (batch_size, dim_fuck)
	A2_mul = merge([a2_con,atten_out], mode='mul')
	A3_mul = merge([a3_con,atten_out], mode='mul')
	A4_mul = merge([a4_con,atten_out], mode='mul')
	A5_mul = merge([a5_con,atten_out], mode='mul')
	
	A1_out = Lambda(sum_one,sum_one_output_shape)(A1_mul)
	A2_out = Lambda(sum_one,sum_one_output_shape)(A2_mul)
	A3_out = Lambda(sum_one,sum_one_output_shape)(A3_mul)
	A4_out = Lambda(sum_one,sum_one_output_shape)(A4_mul)
	A5_out = Lambda(sum_one,sum_one_output_shape)(A5_mul)

	merge_out = merge([A1_out,A2_out,A3_out,A4_out,A5_out],mode='concat')
	
	final_out = Activation('softmax',name='final_out')(merge_out)
	model = Model(input=[ques_input,pass_input,A1_input,A2_input,A3_input,A4_input,A5_input],output=[final_out])
	
	print "Compiling model..."
        rmsprop = RMSprop(lr=args.lr)
	model.compile(loss={'final_out':"categorical_crossentropy"}, optimizer=rmsprop,metrics=['accuracy'])
	print "Compilation done..."
	
	return model

def oneGRUmodel(args, maxlen):
	maxlen_pass = maxlen
	dim_gru = args.dim_gru
	dim_glove = 300
	shared_GRU =  GRU(output_dim = dim_gru, dropout_W=args.dropout, return_sequences = True, input_shape = (maxlen,dim_glove), init = 'glorot_uniform', inner_init = 'orthogonal', inner_activation = 'hard_sigmoid')
	shared_backGRU = GRU(output_dim = dim_gru,dropout_W=args.dropout ,go_backwards=True, return_sequences = True, input_shape = (maxlen,dim_glove), init = 'glorot_uniform', inner_init = 'orthogonal', inner_activation = 'hard_sigmoid')
	pass_input = Input(shape=(maxlen_pass,dim_glove), dtype='float32', name='pass_input')
	pass_gru = shared_GRU(pass_input)
	pass_backgru = shared_backGRU(pass_input)
	pass_con = merge([pass_gru,pass_backgru],mode='concat') # maxlen_pass, 2*dim_gru
#	pass_con = Lambda(the_l2, the_l2_output_shape)(pass_con)

	ques_input = Input(shape=(maxlen,dim_glove), dtype='float32', name='ques_input')
	ques_gru = shared_GRU(ques_input)
	ques_backgru = shared_backGRU(ques_input)
	ques_gru_last = Lambda(get_time_slice,get_time_slice_output_shape)(ques_gru)
	ques_backgru_last = Lambda(get_time_slice,get_time_slice_output_shape)(ques_backgru)
	ques_con = merge([ques_gru_last,ques_backgru_last],mode='concat') # , 2*dim_gru
#	ques_con = Lambda(the_l2, the_l2_output_shape)(ques_con)

	repeat_ques = RepeatVector(maxlen_pass)(ques_con)
	mul_ques_pass = merge([pass_con,repeat_ques],mode='mul') # maxlen_pass, 2*dim_gru
	permute_qp_mul = Permute((2,1))(mul_ques_pass) # 2*dim_gru, maxlen_pass
	#cos_ques_pass = Lambda(cos_sim_matvec,cos_sim_matvec_output_shape)([pass_con,ques_con]) # b_s, maxlen_pass
	dot_ques_pass = Lambda(sum_along_time,sum_along_time_output_shape)(permute_qp_mul) # maxlen_pass
	thecoef = Activation(args.atten_mode)(dot_ques_pass)
	repeat_coeff = RepeatVector(2*dim_gru)(thecoef) # 2*dim_gru, maxlen_pass
	permute_coeff = Permute((2,1))(repeat_coeff) # maxlen_pass, 2*dim_gru
	weighted_vec = merge([permute_coeff, pass_con],mode='mul') # maxlen_pass, 2*dim_gru
	atten_out = Lambda(sum_along_time,sum_along_time_output_shape)(weighted_vec) # 2*dim_gru
#	atten_out = Lambda(the_l2, the_l2_output_shape)(atten_out)

	A1_input = Input(shape=(maxlen,dim_glove),name='A1_input',dtype='float32') # dim_glove
	A2_input = Input(shape=(maxlen,dim_glove),name='A2_input',dtype='float32') # dim_glove
	A3_input = Input(shape=(maxlen,dim_glove),name='A3_input',dtype='float32') # dim_glove
	A4_input = Input(shape=(maxlen,dim_glove),name='A4_input',dtype='float32') # dim_glove
	A5_input = Input(shape=(maxlen,dim_glove),name='A5_input',dtype='float32') # dim_glove

	
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

	a5_gru = shared_GRU(A5_input)
	a5_backgru = shared_backGRU(A5_input)
	a5_gru_last = Lambda(get_time_slice,get_time_slice_output_shape)(a5_gru)
	a5_backgru_last = Lambda(get_time_slice,get_time_slice_output_shape)(a5_backgru)
	a5_con = merge([a5_gru_last,a5_backgru_last],mode='concat') # , 2*dim_gru
#	a1_con = Lambda(the_l2, the_l2_output_shape)(a1_con)
#	a2_con = Lambda(the_l2, the_l2_output_shape)(a2_con)
#	a3_con = Lambda(the_l2, the_l2_output_shape)(a3_con)
#	a4_con = Lambda(the_l2, the_l2_output_shape)(a4_con)


	#A1_mul = Lambda(cos_sim,output_shape=cos_sim_output_shape)([a1_con,atten_out])  # (batch_size, dim_fuck)
	#A2_mul = Lambda(cos_sim,output_shape=cos_sim_output_shape)([a2_con,atten_out])  # (batch_size, dim_fuck)
	#A3_mul = Lambda(cos_sim,output_shape=cos_sim_output_shape)([a3_con,atten_out])  # (batch_size, dim_fuck)
	#A4_mul = Lambda(cos_sim,output_shape=cos_sim_output_shape)([a4_con,atten_out])  # (batch_size, dim_fuck)

	for i in range(args.hop):
		add = merge([atten_out,ques_con],mode='sum')
		ques_con = Lambda(lambda x: x/2)(add)
#		ques_con = Lambda(the_l2, the_l2_output_shape)(ques_con)
		repeat_ques = RepeatVector(maxlen_pass)(ques_con)
		mul_ques_pass = merge([pass_con,repeat_ques],mode='mul') # maxlen_pass, 2*dim_gru
		permute_qp_mul = Permute((2,1))(mul_ques_pass) # 2*dim_gru, maxlen_pass
		#cos_ques_pass = Lambda(cos_sim_matvec,cos_sim_matvec_output_shape)([pass_con,ques_con]) # b_s, maxlen_pass
		dot_ques_pass = Lambda(sum_along_time,sum_along_time_output_shape)(permute_qp_mul) # maxlen_pass
		thecoef = Activation(args.atten_mode)(dot_ques_pass)
		repeat_coeff = RepeatVector(2*dim_gru)(thecoef) # 2*dim_gru, maxlen_pass
		permute_coeff = Permute((2,1))(repeat_coeff) # maxlen_pass, 2*dim_gru
		weighted_vec = merge([permute_coeff, pass_con],mode='mul') # maxlen_pass, 2*dim_gru
		atten_out = Lambda(sum_along_time,sum_along_time_output_shape)(weighted_vec) # 2*dim_gru
#		atten_out = Lambda(the_l2,the_l2_output_shape)(atten_out)

	A1_mul = merge([a1_con,atten_out], mode='mul')  # (batch_size, dim_fuck)
	A2_mul = merge([a2_con,atten_out], mode='mul')
	A3_mul = merge([a3_con,atten_out], mode='mul')
	A4_mul = merge([a4_con,atten_out], mode='mul')
	A5_mul = merge([a5_con,atten_out], mode='mul')
	
	A1_out = Lambda(sum_one,sum_one_output_shape)(A1_mul)
	A2_out = Lambda(sum_one,sum_one_output_shape)(A2_mul)
	A3_out = Lambda(sum_one,sum_one_output_shape)(A3_mul)
	A4_out = Lambda(sum_one,sum_one_output_shape)(A4_mul)
	A5_out = Lambda(sum_one,sum_one_output_shape)(A5_mul)

	merge_out = merge([A1_out,A2_out,A3_out,A4_out,A5_out],mode='concat')
	#final_out = merge([A1_mul,A2_mul,A3_mul,A4_mul],mode='concat',name='final_out')
	
	final_out = Activation('softmax',name='final_out')(merge_out)
	model = Model(input=[ques_input,pass_input,A1_input,A2_input,A3_input,A4_input,A5_input],output=[final_out])
	#model = Model(input=[ques_input,pass_input],output=[final_out])
	
	print "Compiling model..."
        rmsprop = RMSprop(lr=args.lr)
        #rmsprop = RMSprop(lr=0.0001)
	model.compile(loss={'final_out':"categorical_crossentropy"}, optimizer=rmsprop,metrics=['accuracy'])
	print "Compilation done..."

	return model

def loadMQAPickle(path):
	#training_data = cPickle.load(open("Pickle/mc500.train.lstm.noStopWord.pickle"))
	f = open(path, "rb")
	num_data = cPickle.load(f)
	maxlen_pass = cPickle.load(f)
	maxlen_ques = cPickle.load(f)
	maxlen = cPickle.load(f)
	valid_data =  cPickle.load(f)
#	valid_data = cPickle.load(open("Pickle/val.plot.lstm.pickle"))
	f.close()
	return valid_data

def loadMQAMemmap(path,shape1,shape2):
	fp = np.memmap(path, dtype='float32', mode='r+', shape=(shape1,shape2))
	return fp

def loadMCPickle(args):
	training_data = cPickle.load(open("Pickle/mc500.train.lstm.noStopWord.pickle"))
	valid_data = cPickle.load(open("Pickle/mc500.dev.lstm.noStopWord.pickle"))
	if(args.dataset==".x24"):
		training_data = cPickle.load(open("Pickle/mc500.train.lstm.x24.noStopWord.pickle"))
		valid_data = cPickle.load(open("Pickle/mc500.dev.lstm.noStopWord.pickle"))
	elif(args.dataset=="mc500+mc160"):
		training_data = cPickle.load(open("Pickle/mc500+mc160.train.lstm.noStopWord.pickle"))
		valid_data = cPickle.load(open("Pickle/mc160.dev.lstm.noStopWord.pickle"))
	elif(args.dataset=="w=5"):
		training_data = cPickle.load(open("Pickle/mc500+mc160.train.lstm.w=5.pickle"))
		valid_data = cPickle.load(open("Pickle/mc500+mc160.dev.lstm.w=5.pickle"))
	elif(args.dataset=="w=5.mc500"):
		training_data = cPickle.load(open("Pickle/mc500.train.lstm.w=5.pickle"))
		valid_data = cPickle.load(open("Pickle/mc500.dev.lstm.w=5.pickle"))
	elif(args.dataset=="w=5.mc160"):
		training_data = cPickle.load(open("Pickle/mc160.train.lstm.w=5.pickle"))
		valid_data = cPickle.load(open("Pickle/mc160.dev.lstm.w=5.pickle"))
	elif(args.dataset=="cos>0.75.mc160"):
		training_data = cPickle.load(open("Pickle/mc160.train.lstm.cos>0.75.pickle"))
		valid_data = cPickle.load(open("Pickle/mc160.dev.lstm.cos>0.75.pickle"))
	elif(args.dataset=="w=5.mc160.x24"):
		training_data = cPickle.load(open("Pickle/mc160.train.lstm.w=5.x24.pickle"))
		valid_data = cPickle.load(open("Pickle/mc160.dev.lstm.w=5.pickle"))
	elif(args.dataset=="w=10"):
		training_data = cPickle.load(open("Pickle/mc500+mc160.train.lstm.w=10.pickle"))
		valid_data = cPickle.load(open("Pickle/mc500+mc160.dev.lstm.w=10.pickle"))
	elif(args.dataset=="cos>0.75"):
		training_data = cPickle.load(open("Pickle/mc500+mc160.train.lstm.cos>0.75.pickle"))
		valid_data = cPickle.load(open("Pickle/mc500+mc160.dev.lstm.cos>0.75.pickle"))
	else:
		print "Using default dataset argument!!"
	return training_data, valid_data
