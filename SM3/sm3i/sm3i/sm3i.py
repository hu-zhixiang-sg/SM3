from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
from tensorflow.python.util.tf_export import tf_export

import numpy as np

import tensorflow as tf

@tf_export(v1=["train.SM3IOptimizer"])
class SM3IOptimizer(tf.train.Optimizer):
	"""
	@@__init__
	"""
	def __init__(self, learning_rate, momentum, name="SM3I"):
		super(SM3IOptimizer, self).__init__(False, name)
		self._learning_rate = learning_rate
		self._momentum = momentum	
	
	def _create_slots(self, var_list):
		for v in var_list:
			with tf.colocate_with(v):
				if self._momentum > 0:
					self._zeros_slot(v, "momentum", self._name)
				shape = np.array(v.get_shape())
				var_rank = len(shape)
				if var_rank > 1:
					for i, d in enumerate(shape):
						d_tensor = tf.convert_to_tensor(d)
						diag_init = tf.zeros([d_tensor])
						_ = self._get_or_make_slot(v, diag_int, "accumulator_" + str(i), self._name)
				else:
					_ = self._zeros_slot(v, "accumulator", self._name)
		

	def _prepare(self):
		learning_rate = self._call_if_callable(self._learning_rate)
		self._learning_rate_tensor = tf.convert_to_tensor(
			learning_rate, name="learning_rate")
		momentum = self._call_if_callable(self._momentum)
		self._momentum_tensor = tf.convert_to_tensor(momentum, name="momentum")
	
	def _shape_for_broadcasting(self, shape, i):
		rank = len(shape)
		return [1] * i + [shape[i]] + [1] * (rank - i - 1)

	def _compute_past_accumulator(self, accumulators, shape):
		rank = len(shape)
		accumulators_for_broadcasting = [
			tf.reshape(accumulators[i], self._shape_for_broadcasting(shape, i))
			for i in range(rank)
		]
		result = accumulators_for_broadcasting[0]
		"""
		iterating through all possible r's, 
		SM3-ii
		calculating the value of
		min r:S_r{i} mu_{t} + g^2(i)
		SM3-i
		
		"""
		for i in range(1, rank):
			result = tf.minimum(result, accumulators_for_broadcasting[i])
		return result

	def _apply_dense(self, grad, var):
		shape = np.array(var.get_shape())
		var_rank = len(shape)
		if var_rank > 1:
			accumulator_list = [
				self.get_slot(var, "accumulator_" + str(i)) for i in range(var_rank)
			]
			accumulator = self._compute_past_accumulator(accumulator_list, shape, grad)
			"""accumulator += grad * grad"""
		else:
			accumulator_var = self.get_slot(var, "accumulator")
			accumulator = tf.assign_add(accumulator_var , 0)
		""" + 1e-30 for precision? """
		accumulator_inv_sqrt = tf.rsqrt(accumulator + 1e-30)
		scaled_g = (1.0 - self._momentum_tensor) * grad * accumulator_inv_sqrt
		accumulator_update_ops = []
			
		with tf.control_dependencies([g]):
			if var_rank > 1:
				for i, accumulator_i in enumerate(accumulator_list):
					axes = list(range(i)) + list(range(i + 1, var_rank))
					""" computes maximum accross given axis """
					""" for all r: S_r{i} """
					""" mu_t(r) = max{mu_t(r), ne_t(i)} """
					new_accumulator_i = tf.reduce_max(accumulator, axis=axes)
					accumulator_update_ops.append(
						tf.assign(accumulator_i, new_accumulator_i))
		
		with tf.control_dependencies(accumulator_update_ops):
			if self._momentum > 0:
				gbar = self.get_slot(var, "momentum")
				update = tf.assign_add(gbar,
							gbar * (self._momentum_tensor - 1.0) + scaled_g)
			else:
				update = scaled_g
			return tf.assign_sub(var, self._learning_rate_tensor * update)

	def _apply_sparse_shared(self, grad_values, grad_indicies, var):
		shape = np.array(var.get_shape())
		var_rank = len(shape)
		""" """ 
		


	def _resource_apply_dense(self, grad, var):	
		return self._apply_dense(grad, var)


	def _resource_apply_sparse(self, grad_values, var, grad_indices):
		return self._apply_sparse_shared(grad_values, grad_indices, var)


	def _apply_sparse(self, grad, var):
		return self._apply_sparse_shared(grad.values, grad.indices, var)


