import numpy as np

class LinearNode():
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
        self.m, self.n = self.W.shape

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
        # J_ya is a row vector
        assert J_ya.shape == (self.batch_size, 1, self.m)
        self.J_ya = J_ya

        # update the first derivatives of our params
        self.J_y["W"] = J_ya.transpose([0,2,1])@ self.x.transpose([0,2,1])
        self.J_y["b"] = J_ya.T # np.sum(J_ya.T, keepdims=True, axis=1)

        self.update_J_ax()
        self.J_yx = J_ya @ self.J_ax

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


        # update the second derivatives of our params
        H_yW = H_ya.reshape((-1, m, 1, m, 1)) * self.x.reshape((-1, 1, n, 1, 1)) * self.x.reshape((-1, 1, 1, 1, n))
        self.H_y["W"] = H_yW
        self.H_y["b"] = H_ya

        return self.H_yx


