# Ktorch: PyTorch Backend for Keras

##### Note : As of now, there is no integration with PyTorch. This is simply a template for accomodating both imperative and symbolic programming.
------

## Examples

### Imperative

```python
from ktorch import *
import numpy as np

a = Variable(np.zeros((2, 3, 4)))
b = Variable(np.ones((3, 4)))
c = a + 0.2 + b * 0.3
print c
'''
<ktorch.graph.tensor.Tensor object at 0x0000000003E82DA0>
'''
print c.value
'''
[[[ 0.5  0.5  0.5  0.5]
  [ 0.5  0.5  0.5  0.5]
  [ 0.5  0.5  0.5  0.5]]

 [[ 0.5  0.5  0.5  0.5]
  [ 0.5  0.5  0.5  0.5]
  [ 0.5  0.5  0.5  0.5]]]
'''

```

### Symbolic

```python
from ktorch import *
import numpy as np

a = Tensor()
b = Tensor()
c = a + 0.2 + b * 0.3
f = Function([a, b], c)

x = np.zeros((2, 3, 4))
y = np.ones((3, 4))

print f([x, y])[0]  # Function returns a list

'''
[[[ 0.5  0.5  0.5  0.5]
  [ 0.5  0.5  0.5  0.5]
  [ 0.5  0.5  0.5  0.5]]

 [[ 0.5  0.5  0.5  0.5]
  [ 0.5  0.5  0.5  0.5]
  [ 0.5  0.5  0.5  0.5]]]
'''
```


Note that evaluation is greedy. The value of a tensor is computed the instant all the information required to compute it is available. The value will be cached in the `.value` attribute of the tensor. You can explicitly set the value for an input tensor using the `.set_value()` method, and all the tensors in the graph depending on that input will be updated in real time.

```python
from ktorch import *
import numpy as np

a = Tensor()
b = Tensor()
c = Tensor()
d = a + b * c
print d.value
'''Obviously, because we haven't set values for a, b and c
AttributeError: 'Tensor' object has no attribute 'value'
'''
a.set_value(5)
b.set_value(3)
c.set_value(2)
print d.value
'''
11
'''
'''
Change the value for any of the inputs, and value of d will be automatically updated:
'''
c.set_value(4)
print d.value
'''
17
'''
```



