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

        self.update_J_ax()
        self.J_yx = J_ya @ self.J_ax

        return self.J_yx

    def update_J_ax(self):
        self.J_ax = self.W.T

    def update_H_ax(self):
        # this is a linear mapping
        # the second derivative is 0
        self.H_ax = np.zeros((self.m, self.n, self.n))

    def backward_pass2(self, H_ya):

        m,n = self.m, self.n

        # computing the first part of H_ax
        # moving the indices h and k of H_ya to the first two axes
        # creating two versions of J_ax that can be broadcast together
        # after the broadcast, the indices of a will be the first two axes
        # and the indices of x will be the last two
        H_ya = H_ya.reshape((m, m, 1, 1))
        J_ax_1 = self.J_ax.reshape((m, 1, n, 1))
        J_ax_2 = self.J_ax.reshape((1, m, 1, n))

        # summing over the first two axes
        I_yx = (H_ya * J_ax_1 * J_ax_2).sum(axis=0).sum(axis=0)

        # computing the Hessian H_ax
        self.update_H_ax()

        # computing the second part of H_ax
        J_ya_1 = self.J_ya.reshape((m, 1, 1))
        O_yx = (J_ya_1 * self.H_ax).sum(axis=0)

        self.H_yx = I_yx + O_yx
        return self.H_yx


