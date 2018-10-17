import numpy as np


class Node():
    def __init__(self, m, n):
        # x has n entries, a has m entries
        # m is the output dimension, n is the input dimension
        self.m, self.n = m, n

    def forward(self, x):
        # specific implementation has to be done by childnode
        pass

    def backward(self, J_ya):
        self.J_ya = J_ya

        # update the first derivatives of our params
        self.J_y["W"] = np.dot(J_ya.T, self.inp.T)
        self.J_y["b"] = J_ya  # np.sum(J_ya.T, keepdims=True, axis=1)

        self.J_ax = self.W.T
        self.J_yx = J_ya @ self.J_ax

        return self.J_yx

    def backward_pass2(self, H_ya):
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
        # TODO: ...

        # computing the second part of H_ax
        J_ya_1