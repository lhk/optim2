import numpy as np

class LinearNode():
    def __init__(self, W, b):
        self.W = W
        self.b = b

        # x has n entries, a has m entries
        # m is the output dimension, n is the input dimension
        self.m, self.n = self.W.shape

        self.J_y = {}
        self.H_y = {}

    def forward(self, x):
        self.x = x
        self.a = self.W @ x + self.b
        return self.a

    def backward(self, J_ya):
        self.J_ya = J_ya

        # update the first derivatives of our params
        self.J_y["W"] = np.dot(J_ya.T, self.inp.T)
        self.J_y["b"] =J_ya # np.sum(J_ya.T, keepdims=True, axis=1)

        self.J_ax = self.W.T
        self.J_yx = J_ya @ self.J_ax

        return self.J_yx

    def backward_pass2(self, H_ya):

        # computing the first part of H_ax
        # moving the indices h and k of H_ya to the first two axes
        # moving the indices j and j of I_yx to the last two axes
        H_ya = H_ya.reshape((m, m, 1, 1))
        J_ax_1 = self.J_ax.reshape((m, 1, n, 1))
        J_ax_2 = self.J_ax.reshape((1, m, 1, n))

        # summing over the first two axes
        I_yx = (H_ya * J_ax_1 * J_ax_2).sum(axis=0).sum(axis=0)

        # computing the Hessian H_ax
        # this is a 3d tensor, since our layer maps a vector to a vector
        return self.ddy_ddin


