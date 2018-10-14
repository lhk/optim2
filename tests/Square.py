import numpy as np
import unittest

from LinearNode import LinearNode
from SquareNode import SquareNode

class SquareTest(unittest.TestCase):


    def setUp(self):
        self.tol = 1e-6
        self.eps = 1e-6
        np.random.seed(0)

        self.x = np.random.rand(3, 1)

        self.sq = SquareNode()

    def forward(self, x):
        return self.sq.forward(x)

    def backward(self, dy, ddy):
        return self.sq.backward(dy, ddy)

    def test_forward_pass(self):
        # just running forward once, to see if dimensions work
        self.forward(self.x)

    def test_backward(self):
        # just running backward once, to see if dimensions work
        self.forward(self.x)
        self.backward(1, 1)

    def test_grad_x(self):
        for idx in range(self.x.shape[0]):
            mask = np.zeros_like(self.x)
            mask[idx]=1

            x_plus = self.x + mask * self.eps
            x_minus = self.x - mask * self.eps

            a = self.forward(self.x)
            da, dda = 2*a, 2
            dx, ddx = self.backward(da, dda)

            a_plus = self.forward(x_plus)
            a_minus = self.forward(x_minus)

            num_grad = (a_plus ** 2 - a_minus ** 2) / (2 * self.eps)
            num_2grad = (a_plus ** 2 - 2 * a ** 2 + a_minus ** 2) / (self.eps ** 2)

            self.assertTrue(abs(num_grad - dx[idx])<self.tol)
            self.assertTrue(abs(num_2grad - ddx[idx]<np.sqrt(self.tol)))