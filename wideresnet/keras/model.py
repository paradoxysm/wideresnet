# Authors: Jeffrey Wang
# License: BSD 3 clause

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Add
from tensorflow.keras.layers import Dense, Conv2D, ZeroPadding2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.image import stateless_random_flip_left_right, stateless_random_crop

from wideresnet.keras import KaimingNormal, SGDtorch

class WideResNet:
	def __init__(self, input_shape=(32,32,3), n_classes=10,
					depth=16, width=8, lr=0.01, lr_decay=1,
					schedule=None, dropout=0.4, weight_decay=0.0005,
					epochs=100, batch_size=128, preprocess_method=None,
					logging=False, seed=None):
		self.input_shape = input_shape
		self.n_classes = n_classes
		self.depth = depth
		self.width = width
		self.lr = lr
		self.lr_decay = lr_decay
		self.schedule = [] if schedule is None else list(map(int, schedule.split("|")))
		self.dropout = dropout
		self.weight_decay = weight_decay
		self.epochs = epochs
		self.batch_size = batch_size
		self.preprocess_method = preprocess_method
		self.np_rng = np.random.RandomState(seed=seed)
		self.tf_rng = tf.random.Generator.from_seed(self._randint())
		self.model = self._wideresnet()

	def fit(self, data, val=None):
		ds = self.preprocess(data, train=True)
		ds = ds.shuffle(-1).batch(self.batch_size)
		if val is not None:
			val = self.preprocess(val).batch(self.batch_size)
		callbacks = [self._set_lr_schedule()]
		self.model.fit(ds, epochs=self.epochs, validation_data=val, callbacks=callbacks)
		return self

	def preprocess(self, ds, method=None, train=False):
		ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
		if self.preprocess_method == 'cifar10':
			mean = np.array([125.3, 123.0, 113.9]) / 255.0
			std = np.array([63.0, 62.1, 66.7]) / 255.0
			ds =  ds.map(lambda x, y: ((x - mean) / std, y))
			if train:
				p = tf.constant([[4,4],[4,4],[0,0]])
				ds = ds.map(lambda x, y: (tf.pad(x, p, "REFLECT"), y))
				def flip(x, y):
					seed = self.tf_rng.make_seeds(2)[0]
					return stateless_random_flip_left_right(x, seed), y
				ds = ds.map(flip)
				s = tf.constant(self.input_shape)
				def crop(x, y):
					seed = self.tf_rng.make_seeds(2)[0]
					return stateless_random_crop(x, s, seed), y
				ds = ds.map(crop)
		return ds

	def _randint(self):
		return self.np_rng.randint(0, 2**16)

	def _wideresnet(self):
		assert ((self.depth - 4) % 6 == 0)
		n = (self.depth - 4) // 6
		n_stages = [16, 16*self.width, 32*self.width, 64*self.width]
		def weight_init():
			return KaimingNormal(seed=self._randint())

		def conv(net, n, size=(1,1), stride=(1,1), padding=(1,1)):
			if padding is not None:
				net = ZeroPadding2D(padding=padding)(net)
			return Conv2D(n, size, strides=stride, padding='valid',
						kernel_initializer=weight_init(),
						use_bias=False)(net)

		def bn(net):
			return BatchNormalization(momentum=0.9, epsilon=1e-5,
						gamma_initializer='uniform')(net)

		def wideresnet_unit(net, n, stride=(1,1), shortcut=False):
			unit = bn(net)
			unit = Activation('relu')(unit)
			conv_a = conv(unit, n, size=(3,3), stride=stride)
			conv_a = bn(conv_a)
			conv_a = Activation('relu')(conv_a)
			if self.dropout > 0:
				conv_a = Dropout(self.dropout, seed=self._randint())(conv_a)
			conv_a = conv(conv_a, n, size=(3,3), stride=(1,1))
			if shortcut:
				conv_b = conv(unit, n, size=(1,1), stride=stride, padding=None)
			else : conv_b = net
			unit = Add()([conv_a, conv_b])
			return unit

		def wideresnet_stack(net, n, height, stride):
			stack = wideresnet_unit(net, n, stride, shortcut=True)
			for i in range(height-1):
				stack = wideresnet_unit(stack, n)
			return stack

		in_layer = Input(shape=self.input_shape)
		conv1 = conv(in_layer, n_stages[0], size=(3,3), stride=(1,1))
		conv2 = wideresnet_stack(conv1, n_stages[1], n, (1,1))
		conv3 = wideresnet_stack(conv2, n_stages[2], n, (2,2))
		conv4 = wideresnet_stack(conv3, n_stages[3], n, (2,2))
		wrn = bn(conv4)
		wrn = Activation('relu')(wrn)
		wrn = AveragePooling2D(pool_size=(8,8), strides=(1,1), padding='valid')(wrn)
		wrn = Flatten()(wrn)
		wrn = Dense(self.n_classes, kernel_initializer=weight_init())(wrn)
		wrn = Activation('softmax')(wrn)
		model = Model(inputs=in_layer, outputs=wrn)
		sgd = SGDtorch(lr=self.lr, momentum=0.9, nesterov=True, weight_decay=self.weight_decay)
		model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
		return model

	def _set_lr_schedule(self):
		def lr_schedule(epoch_idx, lr):
			if epoch_idx in self.schedule:
				lr *= self.lr_decay
			return lr
		return LearningRateScheduler(lr_schedule)
