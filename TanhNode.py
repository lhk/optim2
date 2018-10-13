import numpy as np

class TanhNode:
    """
    this is just a template
    """
    def __init__(self):
        #no params
        pass

    def forward(self, x):
        """
        Computes the forward pass.
        :param x:
        :return:
        """
        self.x= x
        return np.tanh(x)


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
        dx = dy * 1/(np.cosh(self.x)**2)

        # second derivative of output
        dd_in = ddy * (1/(np.cosh(self.x)**2))**2
        dd_own = dy * -(2*np.sinh(self.x))/(np.cosh(self.x)**3)
        ddx = dd_in + dd_own

        return dx, ddx




