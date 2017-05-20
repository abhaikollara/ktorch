import numpy as np
import torch
import keras
from .graph import *



def variable(value, dtype=None, name=None):
    if dtype is None:
        dtype = keras.backend.floatx()
    if value.dtype.name != dtype:
        value = np.cast[dtype](value)
    torch_tensor = torch.from_numpy(value)
    torch_variable = torch.autograd.Variable(torch_tensor, requires_grad=True)
    ktorch_variable = Variable(torch_variable, name=name)
    return ktorch_variable


def constant(value, dtype=None, name=None):
    if dtype is None:
        dtype = keras.backend.floatx()
    if value.dtype != dtype:
        value = np.cast[dtype](value)
    torch_tensor = torch.from_numpy(value)
    torch_variable = torch.autograd.Variable(torch_tensor, requires_grad=False)
    ktorch_variable = Variable(torch_variable, name=name)
    return ktorch_variable


def placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None):
    if sparse:
        raise Exception('Sparse tensors are not supported yet :( ')
    if dtype is None:
        dtype = keras.backend.floatx()
    ktorch_tensor = Tensor(name=name)
    ktorch_tensor.ndim = ndim
    ktorch_tensor.ndim = ndim
    ktorch_tensor.dtype = dtype


def shape(x):
    if hasattr(x, 'value'):
        return Variable(tuple(x.value.size()))
    elif hasattr(x, 'shape'):
        return Variable(x.shape)
    else:
        raise Exception('Tensor shape not available.')


def int_shape(x):
    if hasattr(x, 'value'):
        return tuple(x.value.size())
    elif hasattr(x, 'shape'):
        return x.shape
    else:
        raise Exception('Tensor shape not available.')  


def ndim(x):
    x_shape = int_shape(x)
    if x_shape is None:
        return None
    else:
        return len(x_shape)


def dtype(x):
    return x.dtype


def eval(x):
    return x.eval().numpy()

def zeros(shape, dtype=None, name=None):
    if dtype is None:
        dtype = floatx()
    return variable(np.zeros(shape), dtype, name)


def ones(shape, dtype=None, name=None):
    if dtype is None:
        dtype = floatx()
    return variable(np.ones(shape), dtype, name)


def eye(size, dtype=None, name=None):
    if dtype is None:
        dtype = floatx()
    return variable(np.eye(size), dtype, name)


def ones_like(x, dtype=None, name=None):
    y = get_op(lambda x: x * 0. + 1.)(x)
    return y


def ones_like(x, dtype=None, name=None):
    y = get_op(lambda x: x * 0.)(x)
    return y


def identity(x):
    y = get_op(lambda x: x + 0.)(x)
    return y

def count_params(x):
    return np.prod(x.eval().size())


def random_uniform_variable(shape, low, high, dtype=None, name=None):
    return variable(np.random.uniform(low=low, high=high, size=shape),
                    dtype=dtype, name=name)


def random_normal_variable(shape, mean, scale, dtype=None, name=None):
    return variable(np.random.normal(loc=0.0, scale=scale, size=shape),
                    dtype=dtype, name=name)



def cast(x, dtype):
    digs = list(map(str, range(10)))
    while dtype[-1] in digs:
        dtype = dtype[:-1]
    y = get_op(lambda x: getattr(x, dtype)())(x)


# UPDATES OPS


def update(x, new_x):
    return (x, new_x)


def update_add(x, increment):
    return (x, x + increment)


def update_sub(x, decrement):
    return (x, x - decrement)


def moving_average_update(variable, value, momentum):
    return (variable, variable * momentum + value * (1. - momentum))


def dot(x, y):
    def _dot(X):
        x, y = X
        if len(x.size()) == 1 and len(y.size()) == 1:
            return torch.dot(x, y)
        else:
            return torch.mm(x, y)
    return get_op(_dot)([x, y])


def batch_dot(x, y, axis=None):
    pass
