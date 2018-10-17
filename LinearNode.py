import numpy as np

class LinearNode():
    def __init__(self, W, b):
        self.W = W
        self.b = b

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
        # this is a linear layer ddout_ddin = 0
        self.ddy_ddin = (self.dout_din**2)@ddy_ddout

        return self.ddy_ddin


