import numpy as np

class LinearNode:
    """
    this is just a template
    """
    def __init__(self, W, b):
        # store the params
        self.W = W
        self.b = b

        self.param_ids= ["W", "b"]

        # parameters
        self.params={}
        self.params["W"]=W
        self.params["b"]=b

        # gradients
        self.dy_d={}
        self.dy_d["W"]=np.empty_like(W)
        self.dy_d["b"]=np.empty_like(b)

        # second order gradients
        self.ddy_dd={}
        self.ddy_dd["W"]=np.empty_like(W)
        self.ddy_dd["b"]=np.empty_like(b)

    def forward_pass1(self, inp):
        self.inp = inp
        self.out = self.W@inp + self.b
        return self.out

    def forward_pass2(self, din_dx):
        self.din_dx = din_dx

        self.dout_dx = din_dx*self.dout_din
        return self.dout_dx

    def backward_pass1(self, dy_dout):
        self.dy_dout = dy_dout

        # update the first derivatives of our params
        self.dy_d["W"][:] = np.dot(dy_dout, self.inp.T)
        self.dy_d["b"][:] = np.sum(dy_dout, keepdims=True, axis=1)

        self.dout_din = self.W.T
        self.dy_din = self.dout_din @ dy_dout

        return self.dy_din

    def backward_pass2(self, ddy_ddout):
        # this is a linear layer ddout_ddin = 0
        self.ddy_ddin = (self.dout_din**2)@ddy_ddout

        return self.ddy_ddin


