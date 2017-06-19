import numpy as np
import torch
import keras
from ..graph import *

def _is_num(x):
    try:
        float(x)
        return True
    except:
        return False


def _get_shape(x):
    if hasattr(x, 'shape'):
        return x.shape
    if hasattr(x, 'size'):
        return tuple(x.size())
    if hasattr(x, 'value'):
        return x.value.size()
    if _is_num(x):
        return ()
    return None


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
    ktorch_tensor = Tensor(name=name, shape=shape, ndim=ndim, dtype=dtype)
    return ktorch_tensor


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
    x_shape = _get_shape(x)
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


def zeros_like(x, dtype=None, name=None):
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
    return y

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
        x_ndim = len(x.size())
        y_ndim = len(y.size())
        if x_ndim == 2 and y_ndim == 2:
            return torch.mm(x, y)
        if x_ndim == 2 and y_ndim == 1:
            return torch.mv(x, y)
        if x_ndim == 1 and y_ndim == 2:
            return torch.mv(y, x)
        if x_ndim == 1 and y_ndim == 1:
            return torch.dot(x, y)
        else:
            raise Exception('Unsupported tensor ranks for dot operation : ' + str(x_ndim) + ' and ' + str(y_ndim) +'.')

    def _compute_output_shape(X):
        x, y = _get_shape(X[0]), _get_shape(X[1])
        x_ndim = len(x)
        y_ndim = len(y)
        if x_ndim == 2 and y_ndim == 2:
            return (x[0], y[1])
        if x_ndim == 2 and y_ndim == 1:
            return (x[0],)
        if x_ndim == 1 and y_ndim == 2:
            return (y[0],)
        if x_ndim == 1 and y_ndim == 1:
            return (0,)
       
    return get_op(_dot, output_shape=_compute_output_shape)([x, y])


def batch_dot(x, y, axes=None):
    if type(axes) is int:
        axes = (axes, axes)
    def _dot(X):
        x, y = X
        x_shape = x.size()
        y_shape = y.size()
        x_ndim = len(x_shape)
        y_ndim = len(y_shape)
        if x_ndim <= 3 and y_ndim <= 3:
            if x_ndim < 3:
                x_diff = 3 - x_ndim
                for i in range(diff):
                    x = torch.unsqueeze(x, x_ndim + i)
            else:
                x_diff = 0
            if y_ndim < 3:
                y_diff = 3 - y_ndim
                for i in range(diff):
                    y = torch.unsqueeze(y, y_ndim + i)
            else:
                y_diff = 0  
            if axes[0] == 1:
                x = torch.transpose(x, 1, 2)
            elif axes[0] == 2:
                pass
            else:
                raise Exception('Invalid axis : ' + str(axes[0]))
            if axes[1] == 2:
                x = torch.transpose(x, 1, 2)
            # -------TODO--------------#


def transpose(x):
    def _transpose(x):
        dim_order = list(reversed(range(ndim(x))))
        return torch.Tensor.permute(x, *dim_order)

    def _compute_output_shape(X):
        return tuple(reversed(_get_shape(X)))

    return get_op(_transpose, output_shape=_compute_output_shape)(x)


# ELEMENT-WISE OPERATION

def max(x, axis=None, keepdims=False):
    def _max(inputs):
        x, axis, keepdims = inputs
        y = torch.max(x, axis)[0]
        # Since keepdims argument of torch not functional
        return y if keepdims else torch.squeeze(y, axis)

    def _compute_output_shape(inputs):
        x, axis, keepdims = inputs
        if not axis:
            return ()

        shape = list(_get_shape(x))
        if keepdims:
            shape[axis] = 1
        else:
            del shape[axis]

        return tuple(shape)

    return get_op(_max, output_shape=_compute_output_shape)([x, axis, keepdims])


def min(x, axis=None, keepdims=False):
    def _min(inputs):
        x, axis, keepdims = inputs
        y = torch.min(x, axis)[0]
        # Since keepdims argument of torch not functional
        return y if keepdims else torch.squeeze(y, axis)

    def _compute_output_shape(inputs):
        x, axis, keepdims = inputs
        if not axis:
            return ()

        shape = list(_get_shape(x))
        if keepdims:
            shape[axis] = 1
        else:
            del shape[axis]

        return tuple(shape)

    return get_op(_min, output_shape=_compute_output_shape)([x, axis, keepdims])


def sum(x, axis=None, keepdims=False):
    def _sum(inputs):
        x, axis, keepdims = inputs
        y = torch.sum(x, axis)
        # Since keepdims argument of torch not functional
        return y if keepdims else torch.squeeze(y, axis)

    def _compute_output_shape(inputs):
        x, axis, keepdims = inputs
        if not axis:
            return ()

        shape = list(_get_shape(x))
        if keepdims:
            shape[axis] = 1
        else:
            del shape[axis]

        return tuple(shape)

    return get_op(_sum, output_shape=_compute_output_shape)([x, axis, keepdims])
