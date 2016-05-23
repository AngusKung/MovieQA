import theano
from theano import tensor as T

from keras.layers import *
from keras.models import Model
from keras import backend as K

import numpy as np

def get_time_slice(x):
	#print x.ndim # b_s, len, dim
	y=x.dimshuffle((1,0,2)) # len, b_s, dim
	#print y.ndim
	return y[-1] #b_s, dim

def get_time_slice_output_shape(input_shape):
	shape = list(input_shape)
	assert len(shape)==3
	outshape = [None, shape[2]]
	return tuple(outshape)

def sum_along_time(x):
	return K.sum(x,axis=1)
def sum_along_time_output_shape(input_shape):
	shape = list(input_shape)
	assert len(shape)== 3
	outshape = [None, shape[2]]
	return tuple(outshape)

def sum_one(x):
	return K.sum(x,axis=1,keepdims=True)
def sum_one_output_shape(input_shape):
	shape = list(input_shape)
	assert len(shape)==2
	outshape = [None, 1]
	return tuple(outshape)

#def myl2(x,axis):
#	#norm = T.sqrt(T.sum(T.square(x),axis=axis,keepdims=True))
#	return x.norm(2,axis=axis,keepdims=True)

def cos_sim(values):
	x, y = values
	x = K.l2_normalize(x, axis=-1)
	y = K.l2_normalize(y, axis=-1)
	return K.sum(x*y, axis=-1,keepdims=True)

def cos_sim_output_shape(shapes):
	shape1, shape2 = shapes
	return (shape1[0],1)

def cos_sim_matvec(values):
	mat, vec = values
	#mat = K.l2_normalize(mat, axis=-1)
	#vec = K.l2_normalize(vec, axis=-1)
	mat = myl2(mat, axis=-1)
	vec = myl2(vec, axis=-1)
	dodo =  K.batch_dot(mat,vec,axes=[2,1])
	#dodo = dodo.dimshuffle((0,1,'x'))
	#return T.extra_ops.repeat()
	return dodo
	#return K.sum(mat*vec, axis=1, keepdims=True)

def cos_sim_matvec_output_shape(shapes):
	shape1, shape2 = shapes
	input_shape = list(shape1)
	return tuple([None,input_shape[1],1])

def the_l2(x):
	#x = K.l2_normalize(x, axis=-1)
	#return T.square(x)
	#x = (T.sum(T.square(x),axis=-1,keepdims=True))
	fuckeps = 1e-10
	x = T.sqrt(T.sum(T.square(x),axis=-1,keepdims=True)+fuckeps)
	return x
def the_l2_output_shape(input_shape):
	shape = list(input_shape)
	return tuple(shape)

def main():

	input_c = np.reshape([1,1,2,2,3,3,4,4,5,5,6,6],(2,2,3)) # batch, len, dim
	input_a = np.reshape([1,2,3,4,5,6],(2,3))
	input_b = np.reshape([7,8,9,10,11,12],(2,3))

	a = Input(shape=(3,))
	b = Input(shape=(3,))
	c = Input(shape=(2,3))

	concat = merge([a,b], mode='concat', concat_axis=-1)
	dot = merge([a,b], mode='dot', dot_axes=1)
	cos = merge([a,b], mode='cos', dot_axes=1)
	fuck = Lambda(sum_one,sum_one_output_shape)(concat)
	mycos = Lambda(cos_sim,cos_sim_output_shape)([a,b])

	lll = Lambda(the_l2,the_l2_output_shape)(c)

	sum_time = Lambda(sum_along_time,sum_along_time_output_shape)(c)

	get_time = Lambda(get_time_slice,get_time_slice_output_shape)(c)

	mycosmatvec = Lambda(cos_sim_matvec,cos_sim_matvec_output_shape)([c,a]) # b_s, len
	r_coeff = RepeatVector(3)(mycosmatvec) # b_s, dim, len
	coeff = Permute((2,1))(r_coeff) # ,len,dim
	atten_out = merge([coeff,c],mode='mul')

	repeat_a = RepeatVector(2)(a)
	mulmul = merge([c,repeat_a],mode='mul')
	per = Permute((2,1))(mulmul)
	dot_ac = Lambda(sum_along_time,sum_along_time_output_shape)(per)

	dot_ac2 = merge([c,repeat_a],mode='dot',dot_axes=(2,2))
	cos_ac2 = merge([c,repeat_a],mode='cos',dot_axes=(2,2))

	model_sum_time  = Model(input=c,output=sum_time)
	model_fuck = Model(input=[a,b],output=fuck)
	model_concat = Model(input=[a,b],output=concat)
	model_dot = Model(input=[a,b],output=dot)
	model_cos = Model(input=[a,b],output=cos)
	model_myco = Model(input=[a,b],output=mycos)
	model_mymatcos = Model(input=[c,a],output=mycosmatvec)
	model_dot1 = Model(input=[a,c],output=dot_ac)
	model_dot2 = Model(input=[a,c],output=dot_ac2)
	model_cos2 = Model(input=[a,c],output=cos_ac2)
	model_lll = Model(input=c,output=lll)
	model_atten = Model(input=[c,a],output=atten_out)
	model_gettime = Model(input=c,output=get_time)

	model_gettime.compile(optimizer='sgd',loss='mse')
	model_sum_time.compile(optimizer='sgd',loss='mean_squared_error')
	model_concat.compile(optimizer='sgd',loss='categorical_crossentropy')
	model_dot.compile(optimizer='sgd',loss='mean_squared_error')
	model_cos.compile(optimizer='sgd',loss='mean_squared_error')
	model_fuck.compile(optimizer='sgd',loss='mean_squared_error')
	model_myco.compile(optimizer='sgd',loss='mean_squared_error')
	model_mymatcos.compile(optimizer='sgd',loss='mean_squared_error')
	model_dot1.compile(optimizer='sgd',loss='mean_squared_error')
	model_dot2.compile(optimizer='sgd',loss='mean_squared_error')
	model_cos2.compile(optimizer='sgd',loss='mean_squared_error')
	model_lll.compile(optimizer='sgd',loss='mean_squared_error')
	model_atten.compile(optimizer='sgd',loss='mse')

	out_concat = (model_concat.predict([input_a,input_b]))
	out_dot = (model_dot.predict([input_a,input_b]))
	out_dot1 = (model_dot1.predict([input_a,input_c]))
	out_dot2 = (model_dot2.predict([input_a,input_c]))
	out_cos = (model_cos.predict([input_a,input_b]))
	out_cos2 = (model_cos2.predict([input_a,input_c]))
	out_fuck = (model_fuck.predict([input_a,input_b]))
	out_myco = (model_myco.predict([input_a,input_b]))
	out_mymatcos = (model_mymatcos.predict([input_c,input_a]))
	out_sumtime = (model_sum_time.predict(input_c))
	out_lll = model_lll.predict(input_c)
	out_atten = model_atten.predict([input_c,input_a])
	out_gettime = model_gettime.predict(input_c)
	
	return out_lll

if __name__ =="__main__":
	oop = main()

