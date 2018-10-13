import numpy as np

from LinearNode import LinearNode
from TanhNode import TanhNode

np.random.seed(0)

W1 = np.random.rand(4, 3)
b1 = np.random.rand(4, 1)
W2 = np.random.rand(1, 4)
b2 = np.random.rand(1, 1)
x = np.random.rand(3, 1)

lin1 = LinearNode(W1, b1)
tan1 = TanhNode()
lin2 = LinearNode(W2, b2)
tan2 = TanhNode()

print("foward pass")
def forward(x):
    z = lin1.forward(x)
    a = tan1.forward(z)
    z = lin2.forward(a)
    a = tan2.forward(z)
    return a

print("backward pass")

print("numerical grad check, dx")
eps = 0.001
mask = np.zeros_like(x)

idx = 0
mask[idx] = 1

x_plus = x+mask*eps
x_minus = x-mask*eps

a=forward(x)
da, dda = 2*a, 2
def backward(dy, ddy):
    dz, ddz = tan2.backward(dy, ddy)
    da, dda = lin2.backward(dz, ddz)
    dz, ddz = tan1.backward(da, dda)
    dx, ddx = lin1.backward(dz, ddz)

    return dx, ddx
dx, ddx = backward(da, dda)

a_plus = forward(x_plus)
a_minus = forward(x_minus)

num_grad = (a_plus**2 - a_minus**2) / (2*eps)
num_2grad = (a_plus**2 - 2*a**2 +a_minus**2) / (eps**2)

print("comparing gradients")
print("analytical vs numerical (1st): {} - {}".format(dx[idx], num_grad))
print("analytical vs numerical (2nd): {} - {}".format(ddx[idx], num_2grad))

print("after getting numerical grads")