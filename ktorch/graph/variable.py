from tensor import Tensor



_FLOATX = 'float32'

def floatx():
	return _FLOATX

class Variable(Tensor):

	def __init__(self, value, **kwargs):
		self.value = value
		shape = getattr(value, 'shape')
		dtype = getattr(value, 'dtype', type(value))
		if dtype is not None and type(dtype) is not str:
			if hasattr(dtype, 'name'):
				dtype = dtype.name
			else:
				dtype = str(dtype)
		super(Variable, self).__init__(shape=shape, dtype=dtype, **kwargs)

	def eval(self):
		return self.value
