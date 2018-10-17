import numpy as np

class Node():
    def __init__(self, m, n, batch_size):
        self.m = m
        self.n = n
        self.batch_size = batch_size

    def forward(self, x):
        raise NotImplementedError("this must be overwritten")

    def backward_pass1(self, J_ya):
        # J_ya is a row vector
        assert J_ya.shape == (self.batch_size, 1, self.m)
        self.J_ya = J_ya

        self.update_J_ax()
        self.J_yx = J_ya @ self.J_ax

        return self.J_yx

    def update_J_ax(self):
        raise NotImplementedError("this must be overwritten")

    def update_H_ax(self):
        raise NotImplementedError("this must be overwritten")

    def backward_pass2(self, H_ya):

        m,n = self.m, self.n
        batch_size = self.batch_size

        # computing the first part of H_ax
        # moving the indices h and k of H_ya to the first two axes
        # creating two versions of J_ax that can be broadcast together
        # after the broadcast, the indices of a will be the first two axes
        # and the indices of x will be the last two
        H_ya = H_ya.reshape((-1, m, m, 1, 1))
        J_ax_1 = self.J_ax.reshape((-1, m, 1, n, 1))
        J_ax_2 = self.J_ax.reshape((-1, 1, m, 1, n))

        # summing over the first two axes
        I_yx = (H_ya * J_ax_1 * J_ax_2).sum(axis=1).sum(axis=1)

        # computing the Hessian H_ax
        self.update_H_ax()

        # computing the second part of H_ax
        J_ya_1 = self.J_ya.reshape((batch_size, m, 1, 1))
        O_yx = (J_ya_1 * self.H_ax).sum(axis=1)

        self.H_yx = I_yx + O_yx
        return self.H_yx


