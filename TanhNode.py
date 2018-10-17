import numpy as np

class TanhNode:
    def __init__(self):
        #no params
        self.param_ids=[]

    def forward_pass1(self, inp):
        self.inp= inp
        self.out = np.tanh(inp)
        return self.out

    def forward_pass2(self, din_dx):

        self.din_dx = din_dx

        # calculate derivative of output wrt x
        self.dout_dx = din_dx * self.dout_din

        return self.dout_dx


    def backward_pass1(self, dy_dout):
        self.dy_dout = dy_dout
        self.dout_din = 1 / (np.cosh(self.inp) ** 2)
        self.dy_din = dy_dout * self.dout_din

        return self.dy_din

    def backward_pass2(self, ddy_ddout):
        self.ddout_ddin = -(2*np.sinh(self.inp))/(np.cosh(self.inp)**3)

        self.ddy_ddin = ddy_ddout * (self.dout_din)**2 + self.dy_dout * self.ddout_ddin

        return self.ddy_ddin


