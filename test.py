import numpy as np

from LinearNode import LinearNode
from TanhNode import TanhNode

np.random.seed(0)

W = np.random.rand(1,3)
b = np.random.rand(1)
x = np.random.rand(3, 1)

lin = LinearNode(W, b)
tan = TanhNode()

print("foward pass")
z = lin.forward(x)
a = tan.forward(z)

print("backward pass")

print("numerical grad check, dx")
eps = 0.001
mask = np.zeros_like(x)

idx = 0
mask[idx] = 1

x_plus = x+mask*eps
x_minus = x-mask*eps

a=tan.forward(lin.forward(x))
da, dda = 1, 0
dz, ddz = tan.backward(da, dda)
dx, ddx = lin.backward(dz, ddz)

a_plus = tan.forward(lin.forward(x_plus))
a_minus = tan.forward(lin.forward(x_minus))

num_grad = (a_plus - a_minus) / (2*eps)
num_2grad = (a_plus - 2*a +a_minus) / (eps**2)

print("comparing gradients")
print("analytical vs numerical (1st): {} - {}".format(dx[idx], num_grad))
print("analytical vs numerical (2nd): {} - {}".format(ddx[idx], num_2grad))

print("after getting numerical grads")