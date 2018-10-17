import numpy as np

class SigmoidNode:
    def __init__(self, n, batch_size):
        #no params
        self.param_ids=[]

        self.n = n
        self.batch_size=batch_size

    def forward(self, x):

        self.x= x
        self.exp_negx = np.exp(-x)
        self.a = 1/(1+self.exp_negx)
        return self.a

    def backward_pass1(self, J_ya):
        # J_ya is a row vector
        assert J_ya.shape == (self.batch_size, 1, self.n)
        self.J_ya = J_ya

        self.update_J_ax()
        self.J_yx = J_ya @ self.J_ax

        return self.J_yx


    def update_J_ax(self):
        n = self.n
        batch_size = self.batch_size
        self.J_ax = np.zeros((batch_size, n,n))
        self.J_ax[:, np.arange(n), np.arange(n)] = (self.exp_negx/(self.exp_negx +1)**2).reshape((batch_size, n))

    def update_H_ax(self):
        n = self.n
        batch_size = self.batch_size
        self.H_ax = np.zeros((batch_size, n, n, n))
        self.H_ax[:, np.arange(n), np.arange(self.n), np.arange(self.n)] =  (((2*np.exp(-2*self.x)/(self.exp_negx+1)**3) - self.exp_negx/(self.exp_negx +1)**2)).reshape((batch_size, n))


    def backward_pass2(self, H_ya):

        m,n = self.n, self.n
        batch_size = self.batch_size

        # computing the first part of H_ax
        # moving the indices h and k of H_ya to the first two axes
        # creating two versions of J_ax that can be broadcast together
        # after the broadcast, the indices of a will be the first two axes
        # and the indices of x will be the last two
        H_ya = H_ya.reshape((batch_size, m, m, 1, 1))
        J_ax_1 = self.J_ax.reshape((batch_size, m, 1, n, 1))
        J_ax_2 = self.J_ax.reshape((batch_size, 1, m, 1, n))

        # summing over the first two axes (skipping batch_size)
        I_yx = (H_ya * J_ax_1 * J_ax_2).sum(axis=1).sum(axis=1)

        # computing the Hessian H_ax
        self.update_H_ax()

        # computing the second part of H_ax
        J_ya_1 = self.J_ya.reshape((batch_size, m, 1, 1))
        O_yx = (J_ya_1 * self.H_ax).sum(axis=1)

        self.H_yx = I_yx + O_yx
        return self.H_yx