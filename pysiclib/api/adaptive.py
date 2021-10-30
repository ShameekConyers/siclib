from .._pysiclib.adaptive import *
from .. import _pysiclib

import numpy as np
import scipy.special

class ProtoNet_Numpy:


	def __init__(self,
		input_size, hidden_size, hidden_layers, output_size, learning_rate,
		other_net: ProtoNet):


		self.input_size = input_size
		self.hidden_size = hidden_size
		self.hidden_layers = hidden_layers
		self.output_size = output_size
		self.learning_rate = learning_rate

		self.weights = []
		self.bias = []
		# self.func = other_net.m_transform
		# self.func_deriv = other_net.m_transform_deriv
		self.func = lambda x: scipy.special.expit(x)
		for item in other_net.m_weights:
			self.weights.append(item.to_numpy())
			# print(item.to_numpy().shape)
			# print(item.get_shape())

		for item in other_net.m_bias:
			self.bias.append(item.to_numpy())
			# print(item.to_numpy().shape)
			# print(item.get_shape())

		assert(len(self.bias) == 2)

	def run_epoch(self, inp, targ):
		inp = np.array(inp, ndmin= 2).T
		targ = np.array(targ, ndmin= 2).T
		#fwd
		hi: np.array = np.dot(self.weights[0], inp)
		ho = self.func(hi)
		fi = np.dot(self.weights[1], ho)
		fo = self.func(fi)

		oe = targ - fo
		he = np.dot(self.weights[1].T, oe)

		delta = self.learning_rate *\
			np.dot(( oe * fo * (1 - fo)), np.transpose(ho))
		self.weights[1] += delta

		self.weights[0] += self.learning_rate *\
			np.dot(( he * ho * (1 - ho)), np.transpose(inp))


	def query_net(self, inputs_list):
		inputs = np.array(inputs_list, ndmin = 2).T
		hi: np.array = np.dot(self.weights[0], inputs)
		ho = self.func(hi)
		fi = np.dot(self.weights[1], ho)
		fo = self.func(fi)
		return fo
