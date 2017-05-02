from tensor import Tensor

class Variable(Tensor):

	def __init__(self, value):
		self.value = value

	def eval(self):
		return self.value
