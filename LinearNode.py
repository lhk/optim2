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
        self.d={}
        self.d["W"]=np.empty_like(W)
        self.d["b"]=np.empty_like(b)

        # second order gradients
        self.dd={}
        self.dd["W"]=np.empty_like(W)
        self.dd["b"]=np.empty_like(b)

    def forward(self, x):
        """
        Computes the forward pass.
        :param x:
        :return:
        """
        self.x= x
        return self.W@x + self.b


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

        # update the first derivatives of our params
        self.d["W"][:] = np.dot(dy, self.x.T)
        self.d["b"][:] = np.sum(dy, keepdims=True, axis=1)

        # update the second derivatives of our params
        # the incoming second derivs need to be multiplied by our first, squared
        # we don't need to add the square of our own derivative, this is a linear function so it's 0
        self.dd["W"][:] = np.dot(ddy, self.x.T**2)
        self.dd["b"][:] = np.sum(ddy, keepdims=True, axis=1)

        # first derivative of output
        dx = np.dot(self.W.T, dy)

        # second derivative of output
        # again, no contribution of our own second derivative (linear function)
        ddx = np.dot(self.W.T**2, ddy)

        return dx, ddx




