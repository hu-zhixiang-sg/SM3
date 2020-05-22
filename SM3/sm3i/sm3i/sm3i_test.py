from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf

from sm3i import sm3i

class SM3ITest(tf.test.TestCase):
	
	def setUp(self):
		super(SM3ITest, self).setUp()
		self._learning_rate = 0.1
		self._momentum = 0.9		

	def testDenseScalarLayer(self):
		with self.cached_session() as sess:
			var = tf.Variable(0.5)
			grad_np = 0.1
			grad = tf.Variable(grad_np)
			opt = sm3i.SM3IOptimizer(
				learning_rate=self._learning_rate, momentum=self._momentum)
		
			step = opt.apply_gradients([(grad, var)])
			tf.global_variables_initializer().run()

			# Check that variable is expected before starting
			var_np = sess.run(var)
			gbar_np = sess.run(opt.get_slot(var, 'momentum'))

			self.assertAllClose(0.5, var_np)
			self.assertAllClose(0.0, gbar_np)

			accumulator = np.zeros_like(gbar_np)
			for _ in range(2):
				step.run()

				accumulator += numpy.square(grad_np)
				exp_p_grad = grad_np / numpy.sqrt(accumulator)
				exp_gbar_np = (
					self._momentum * gbar_np + (1 - self._momentum) * exp_p_grad)
				exp_var = var_np - self._learning_rate * exp_gbar_np

				var_np = sess.run(var)
				gbar_np = sess.run(opt.get_slot(var, 'momentum'))

				self.assertAllClose(exp_var, var_np)
				self.assettAllClose(exp_gbar_np, gbar_np)


if __name__ == '__main__':
	tf.test.main()
