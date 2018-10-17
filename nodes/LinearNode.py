import numpy as np
from nodes.Node import Node

class LinearNode(Node):
    def __init__(self, W, b, batch_size):
        self.W = W
        self.b = b
        self.batch_size = batch_size

        self.param_ids=["W", "b"]
        self.params={}
        self.params["W"]=W
        self.params["b"]=b

        # x has n entries, a has m entries
        # m is the output dimension, n is the input dimension
        m, n = self.W.shape

        super().__init__(m, n, batch_size)

        self.J_y = {}
        self.J_a = {}

        # W is stored as a matrix
        # this leads to a complicated layout in the hessian:
        # two indices in W select one specific parameter
        # so four indices in the Hessian select the second derivative:
        # the second derivative of y wrt to W_ij and W_hk is
        # H_y[batch, i, j, h, k]
        # alternatively, this matrix could be flattened
        # then H_y would become 3 dimensional: batch and the two variable axes
        self.H_y = {}

    def forward(self, x):
        assert x.shape==(self.batch_size, self.n, 1)

        self.x = x
        self.a = self.W @ x + self.b
        return self.a

    def backward_pass1(self, J_ya):
        super().backward_pass1(J_ya)

        # update the first derivatives of our params
        self.J_y["W"] = J_ya.transpose([0,2,1])@ self.x.transpose([0,2,1])
        self.J_y["b"] = J_ya.transpose([0,2,1]) # np.sum(J_ya.T, keepdims=True, axis=1)

        return self.J_yx

    def update_J_ax(self):
        m, n = self.m, self.n
        self.J_ax = self.W.reshape((1, m, n))

    def update_H_ax(self):
        # this is a linear mapping
        # the second derivative is 0
        self.H_ax = np.zeros((1, self.m, self.n, self.n))

    def backward_pass2(self, H_ya):

        m,n = self.m, self.n
        super().backward_pass2(H_ya)

        # update the second derivatives of our params
        H_yW = H_ya.reshape((-1, m, 1, m, 1)) * self.x.reshape((-1, 1, n, 1, 1)) * self.x.reshape((-1, 1, 1, 1, n))
        self.H_y["W"] = H_yW

        # b has the shape (4,1), we want the resulting Hessian to be indexed along the axes 2 and 4:
        self.H_y["b"] = H_ya.reshape((-1, m, 1, m, 1))

        return self.H_yx


