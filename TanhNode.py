import numpy as np

class TanhNode:
    def __init__(self):
        #no params
        self.param_ids=[]

    def forward(self, inp):
        """
        out = tanh(inp)
        :param inp: inflowing input
        :return:
        """
        self.inp= inp
        self.out = np.tanh(inp)
        return self.out

    def forward_grad(self, din_dx, ddy):

        self.din_dx = din_dx

        # calculate derivative of output wrt x
        self.dout_dx = din_dx * self.dout_din

        # inflowing second times own first
        dd_in = ddy * self.dout_din

        # own second
        ddout_ddin = -(2*np.sinh(self.x))/(np.cosh(self.x)**3)
        dd_own = self.din_dx**2 * ddout_ddin

        ddy = dd_in + dd_own

        return self.dout_dx, ddy



    def backward(self, dy_dout):
        """
        dy_din = dy_dout * dout_din
        :param dy_dout: inflowing first gradient
        :return:
        """
        self.dout_din = 1 / (np.cosh(self.inp) ** 2)
        self.dy_din = dy_dout * self.dout_din

        return self.dy_din




