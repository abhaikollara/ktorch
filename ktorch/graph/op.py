from tensor import Tensor

class Op(object):

	def __init__(self):
		if not hasattr(self, 'num_inputs'):
			self.num_inputs = None

	def call(self, x):
		# imperative code goes here
		return x

	def __call__(self, x):
		if type(x) in [list, tuple]:
			x_len = len(x)
			if len(x) == 1:
				x = x[0]
			else:
				x = list(x)
		else:
			x_len = 1
		self._check_num_inputs(x_len)
		y = Tensor()
		y.op = self
		if type(x) is tuple:
			x = list(x)
		y.inputs = x
		return y

	def _check_num_inputs(self, n):
		if not hasattr(self, 'num_inputs') or self.num_inputs is None:
			return
		class_name = self.__class__.__name__
		num_inputs = self.num_inputs
		if type(num_inputs) is str:
			if '+' in num_inputs:
				min_num_inputs = int(num_inputs[:-1])
				if n < min_num_inputs:
					raise Exception(class_name + ' expected at least ' + str(min_num_inputs) + ' inputs but received only ' + str(n) + ' inputs.')
				return
			else:
				num_inputs = int(num_inputs)
		if num_inputs != n:
		    raise Exception(class_name + ' expected ' + str(num_inputs) + ' inputs but received ' + str(len(x)) +' inputs.')

