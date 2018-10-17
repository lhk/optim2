import numpy as np

class TanhNode:
    def __init__(self, n):
        #no params
        self.param_ids=[]

        self.n = n
        self.m = n

    def forward(self, x):
        self.x = x
        self.a = np.tanh(x)
        return self.a

    def backward(self, J_ya):
        self.J_ya = J_ya

        self.update_J_ax()
        self.J_yx = J_ya @ self.J_ax

        return self.J_yx


    def update_J_ax(self):
        self.J_ax = np.diag(1 / (np.cosh(self.x) ** 2))

    def update_H_ax(self):
        n = self.n
        self.H_ax = np.zeros((n, n, n))
        self.H_ax[np.arange(n), np.arange(self.n), np.arange(self.n)] =  -(2*np.sinh(self.x))/(np.cosh(self.x)**3)


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