from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf

@tf_export(v1=["train.SM3Optimizer"])
class SM3Optimizer(Optimizer.Optimizer):
	"""
	@@__init__
	"""
	def __init__(self, learning_rate, use_locking, nam="SM3"):
		super(SM3, self).__init__(use_locking, name)
		self._learning_rate = learning_rate		
		# do we need alpha, beta, epsilon variables?
	

	def _prepare(self):
		#self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
		#self._alpha_t = ops.convert_to_tensor(self._beta, name="alpha_t")
		#self._beta_t = ops.convert_to_tensor(self.beta, name="beta_t")

	def _create_slots(self, grad, var):
		""" SM3 algo goes here """

	def _apply_sparse(self, grad, var):
		""" """

	def _apply_dense(self, grad, var):
