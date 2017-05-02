from ktorch import *
import numpy as np


x = np.random.random((2, 3, 4))
y = np.random.random((3, 4))

# Imperative

a = Variable(x)
b = Variable(y)

c = a + b * 0.3

c.eval()

assert c[:1, :2].eval().shape == (1, 2, 4)

# Symbolic

a = Tensor()
b = Tensor()

d = a + b * 0.3

f = Function([a, b], d)

assert np.all(f([x, y]) == c.eval())

# Advanced indexing

a = Tensor()
b = a[len(a)]
f = Function(a, b)

assert f(x)[0].shape == (1, 3, 4)
