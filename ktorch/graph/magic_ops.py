from op import Op


class Add(Op):

	def __init__(self):
		self.num_inputs = '2+'
		super(Add, self).__init__()

	def call(self, x):
		z = x[0]
		for i in x[1:]:
			z = z + i
		return z


class Multiply(Op):

	def __init__(self):
		self.num_inputs = '2+'
		super(Multiply, self).__init__()

	def call(self, x):
		z = x[0]
		for i in x[1:]:
			z = z * i
		return z


class Subtract(Op):

	def __init__(self):
		self.num_inputs = 2
		super(Subtract, self).__init__()

	def call(self, x):
		return x[0] - x[1]


class Divide(Op):

	def __init__(self):
		self.num_inputs = 2
		super(Divide, self).__init__()

	def call(self, x):
		return x[0] / x[1]


class IntegerDivide(Op):

	def __init__(self):
		self.num_inputs = 2
		super(IntegerDivide, self).__init__()

	def call(self, x):
		return x[0] // x[1]


class Mod(Op):

	def __init__(self):
		self.num_inputs = 2
		super(Mod, self).__init__()

	def call(self, x):
		return x[0] % x[1]


class Power(Op):

	def __init__(self):
		self.num_inputs = 2
		super(Power, self).__init__()

	def call(self, x):
		return x[0] ** x[1]

class Absolute(Op):

	def __init__(self):
		self.num_inputs = 1
		super(Absolute, self).__init__()

	def call(self, x):
		return abs(x)

class Length(Op):

	def __init__(self):
		self.num_inputs = 1
		super(Length, self).__init__()

	def call(self, x):
		return len(x)

class Negative(Op):

	def __init__(self):
		self.num_inputs = 1
		super(Negative, self).__init__()

	def call(self, x):
		return -x

class Positive(Op):

	def __init__(self):
		self.num_inputs = 1
		super(Positive, self).__init__()

	def call(self, x):
		return +x

class LessThan(Op):

	def __init__(self):
		self.num_inputs = 2
		super(LessThan, self).__init__()

	def call(self, x):
		return x[0] < x[1]

class LessThanOrEqual(Op):

	def __init__(self):
		self.num_inputs = 2
		super(LessThanOrEqual, self).__init__()

	def call(self, x):
		return x[0] <= x[1]

class LessThan(Op):

	def __init__(self):
		self.num_inputs = 2
		super(LessThan, self).__init__()

	def call(self, x):
		return x[0] < x[1]

class LessThanOrEqual(Op):

	def __init__(self):
		self.num_inputs = 2
		super(LessThanOrEqual, self).__init__()

	def call(self, x):
		return x[0] <= x[1]

class GreaterThan(Op):

	def __init__(self):
		self.num_inputs = 2
		super(GreaterThan, self).__init__()

	def call(self, x):
		return x[0] > x[1]

class GreaterThanOrEqual(Op):

	def __init__(self):
		self.num_inputs = 2
		super(GreaterThanOrEqual, self).__init__()

	def call(self, x):
		return x[0] >= x[1]

class Equal(Op):

	def __init__(self):
		self.num_inputs = 2
		super(Equal, self).__init__()

	def call(self, x):
		return x[0] == x[1]

class Slice(Op):

	def __init__(self):
		self.num_inputs	= 2
		super(Slice, self).__init__()

	def call(self, x):
		return x[0][x[1]]


# functional interface
all_ops = [cls for cls in globals().values() if type(cls) is type and issubclass(cls, Op)]
def _to_lower(x):
	y = ''
	for i in range(len(x)):
		c = x[i]
		if c.isupper():
			if i:
				y += '_'
		y += c.lower()
	return y
for op in all_ops:
	op_name = op.__name__
	exec('def ' + _to_lower(op_name) + '(*args):return ' + op_name + '()(args)')
del _to_lower
del all_ops
del op
del op_name


###useful abbrvts.###
lt = less_than
le = less_than_or_equal
gt = greater_than
ge = greater_than_or_equal
eq = equal
