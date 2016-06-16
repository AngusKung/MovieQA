import theano
from theano import tensor as T

import keras
from keras.layers import *
from keras.models import Model
from keras import backend as K

import numpy as np

class CustomTimeDistributedDense(Layer):
    def __init__(self, output_dim,
                 init='glorot_uniform', activation='linear', weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.activation = activations.get(activation)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.initial_weights = weights
        self.supports_masking = True

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(CustomTimeDistributedDense, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[2]

        self.W = self.init((input_dim, self.output_dim),
                           name='{}_W'.format(self.name))
        self.b = keras.backend.zeros((self.output_dim,),
                         name='{}_b'.format(self.name))

        self.trainable_weights = [self.W, self.b]
        self.regularizers = []

        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint
        if self.b_constraint:
            self.constraints[self.b] = self.b_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def call(self, x, mask=None):
        input_shape = x.shape
        # x has shape (samples, timesteps, input_dim)
        input_length = input_shape[1]
        # Note: input_length should always be provided when using tensorflow backend.
        if not input_length:
            if hasattr(keras.backend, 'int_shape'):
                input_length = keras.backend.int_shape(x)[1]
                if not input_length:
                    raise Exception(
                        'Layer ' + self.name +
                        ' requires to know the length of its input, '
                        'but it could not be inferred automatically. '
                        'Specify it manually by passing an input_shape '
                        'argument to the first layer in your model.')
            else:
                input_length = keras.backend.shape(x)[1]

        # Squash samples and timesteps into a single axis
        x = keras.backend.reshape(x, (-1, input_shape[-1]))  # (samples * timesteps, input_dim)
        y = keras.backend.dot(x, self.W) + self.b  # (samples * timesteps, output_dim)
        # We have to reshape Y to (samples, timesteps, output_dim)
        y = keras.backend.reshape(y, (-1, input_length, self.output_dim))  # (samples, timesteps, output_dim)
        y = self.activation(y)
        return y

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'input_dim': self.input_dim,
                  'input_length': self.input_length}
        base_config = super(CustomTimeDistributedDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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

