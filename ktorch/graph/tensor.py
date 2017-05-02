class Tensor(object):

	def eval(self):
		if not hasattr(self, 'value'):
			if not hasattr(self, 'op') or self.op is None:
			    raise Exception('Input tensor was not provided value.')
			if type(self.inputs) is list:
				self.value = self.op.call([evaluate(x) for x in self.inputs])
			else:
				self.value = self.op.call(self.inputs.eval())
		return self.value

	def set_value(self, value):
		self.value = value

	def __add__(self, x):
		from .magic_ops import add
		return add(self, x)

	def __radd__(self, x):
		from .magic_ops import add
		return add(x, self)

	def __mul__(self, x):
		from .magic_ops import multiply
		return multiply(self, x)

	def __rmul__(self, x):
		from .magic_ops import multiply
		return multiply(x, self)

	def __sub__(self, x):
		from .magic_ops import subtract
		return subtract(self, x)

	def __rsub__(self, x):
		from .magic_ops import subtract
		return subtract(x, self)

	def __div__(self, x):
		from .magic_ops import divide
		return divide(self, x)

	def __rdiv__(self, x):
		from .magic_ops import divide
		return divide(x, self)

	def __floordiv__(self, x):
		from .magic_ops import integer_divide
		return integer_divide(self, x)

	def __rfloordiv__(self, x):
		from .magic_ops import integer_divide
		return integer_divide(x, self)

	def __mod__(self, x):
		from .magic_ops import mod
		return mod(self, x)

	def __rmod__(self, x):
		from .magic_ops import mod
		return mod(x, self)

	def __div__(self, x):
		from .magic_ops import divide
		return divide(self, x)

	def __rdiv__(self, x):
		from .magic_ops import divide
		return divide(x, self)

	def __pow__(self, x):
		from .magic_ops import power
		return power(self, x)

	def __rpow__(self, x):
		from .magic_ops import power
		return power(x, self)

	def __neg__(self):
		from.magic_ops import negative
		return negative(self)

	def __pos__(self):
		from.magic_ops import positive
		return positive(self)

	def __abs__(self):
		from.magic_ops import absolute
		return absolute(self)

	def __len__(self):
		from.magic_ops import length
		return length(self)

	def __lt__(self, x):
		from .magic_ops import lt
		return lt(self, x)	

	def __le__(self, x):
		from .magic_ops import le
		return le(self, x)	

	def __gt__(self, x):
		from .magic_ops import gt
		return gt(self, x)

	def __ge__(self, x):
		from .magic_ops import ge
		return ge(self, x)

	def __getitem__(self, x):
		from .magic_ops	import slice
		return slice(self, x)

def evaluate(x):
	if isinstance(x, Tensor):
		return x.eval()
	else:
			x_type = type(x)
			if x_type is list:
				return list(map(evaluate, x))
			elif x_type is tuple:
				return tuple(list(map(evaluate, x))) 
			elif x_type is dict:
				return {evaluate(k):evaluate(x[k]) for k in x}
			elif x_type is slice:
				return slice(*(list(map(evaluate, [x.start, x.stop, x.step]))))
			else:
				return x
