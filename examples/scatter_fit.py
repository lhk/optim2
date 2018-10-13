import numpy as np
import matplotlib.pyplot as plt

# set up the data
# circle one
num_samples=100
phi = np.linspace(0, np.pi * 5, num_samples)
r = np.linspace(0, 10, num_samples)

x1 = np.cos(phi)*r
y1 = np.sin(phi)*r

labels1 = np.zeros((num_samples,))

# circle two
num_samples=100
phi = np.linspace(0 + np.pi/2, np.pi * 5 + np.pi/1.3, num_samples)
r = np.linspace(0, 10, num_samples)

x2 = np.cos(phi)*r
y2 = np.sin(phi)*r

labels2 = np.ones((num_samples,))

x = np.concatenate([x1, x2]).reshape((-1, 1))
y = np.concatenate([y1, y2]).reshape((-1, 1))
labels = np.concatenate([labels1, labels2])

#plt.scatter(x, y, c=labels)
#plt.show()

# set up the network
from LinearNode import LinearNode
from TanhNode import TanhNode
from SigmoidNode import SigmoidNode

np.random.seed(0)

shapes = [10, 100, 1]
input_size = 2
nodes = []
for shape in shapes:
    W = np.random.rand(shape, input_size) * 0.01
    b = np.random.rand(shape, 1) * 0.01
    input_size = shape

    lin = LinearNode(W, b)
    tanh = TanhNode()
    nodes.append(lin)
    nodes.append(tanh)

# remove last tanh
# TODO: this is too hacky, do it nicely
nodes.remove(nodes[-1])

sigm = SigmoidNode()
nodes.append(sigm)

def forward(x):
    for node in nodes:
        x= node.forward(x)
    return x

def backward(dy, ddy):
    for node in reversed(nodes):
        dy, ddy = node.backward(dy, ddy)
    return dy, ddy

def sample(batch_size):
    indices = np.random.choice(200, batch_size)

    batch_x = x[indices]
    batch_y = y[indices]
    batch = np.concatenate([batch_x, batch_y], axis=1).T

    targets = labels[indices]

    return batch, targets

def update_firstorder(alpha=1e-3):
    for node in nodes:
        for param_id in node.param_ids:
            # something like a relaxed newton step
            node.params[param_id][:] = node.params[param_id] - alpha*node.d[param_id]

def update_secondorder(alpha=1e-5):
    for node in nodes:
        for param_id in node.param_ids:
            # something like a relaxed newton step
            node.params[param_id][:] = node.params[param_id] - alpha*node.d[param_id]/node.dd[param_id]


# start the training process
batch_size = 16
for i in range(10000):
    # get a training sample
    batch, targets = sample(batch_size=batch_size)

    # forward pass
    preds = forward(batch)

    # compute and derive loss
    loss = ((preds - targets)**2).mean()
    dy = 2*(preds - targets)
    ddy = 2

    # backward pass
    backward(dy, ddy)

    # update params
    update_firstorder()

    if i%10==0:
        print(loss)

