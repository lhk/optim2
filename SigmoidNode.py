import numpy as np

class SigmoidNode:
    """
    this is just a template
    """
    def __init__(self):
        #no params
        self.param_ids=[]

    def forward(self, x):
        """
        Computes the forward pass.
        :param x:
        :return:
        """
        self.x= x
        self.exp_negx = np.exp(-x)
        return 1/(1+self.exp_negx)


    def backward(self, dy, ddy):
        """
        Computes the backward pass from inflowing gradients.
        Creating the first derivatives is a simple backward pass.
        Calculating the second derivs is harder:
        We already receive a second derivative, this has to be multiplied by the square of our first derivs.
        Add to this our second deriv times the incoming first.

        :param dy: first derivative
        :param ddy: second derivative
        :return:
        """

        # first derivative of output
        dx = dy * self.exp_negx/(self.exp_negx +1)**2

        # second derivative of output
        dd_in = ddy * (self.exp_negx/(self.exp_negx +1)**2)**2
        dd_own = dy * ((2*np.exp(-2*self.x)/(self.exp_negx+1)**3) - self.exp_negx/(self.exp_negx +1)**2)
        ddx = dd_in + dd_own

        return dx, ddx




