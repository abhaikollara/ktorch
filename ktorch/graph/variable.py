from tensor import Tensor

class Variable(Tensor):

	def __init__(self, value):
		self.value = value
		super(Variable, self).__init__()

	def eval(self):
		return self.value
