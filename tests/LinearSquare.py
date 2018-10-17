import numpy as np
import unittest

from LinearNode import LinearNode
from SquareNode import SquareNode

class LinearSquareTest(unittest.TestCase):


    def setUp(self):
        self.tol = 1e-6
        self.eps = 1e-6
        np.random.seed(0)

        W1 = np.random.rand(4, 3)
        b1 = np.random.rand(4, 1)
        W2 = np.random.rand(1, 4)
        b2 = np.random.rand(1, 1)

        self.x = np.random.rand(3, 1)

        self.lin1 = LinearNode(W1, b1)
        self.sq1 = SquareNode()
        self.lin2 = LinearNode(W2, b2)
        self.sq2 = SquareNode()

    def forward(self, x):
        z = self.lin1.forward(x)
        a = self.sq1.forward(z)
        z = self.lin2.forward(a)
        a = self.sq2.forward(z)
        return a

    def backward(self, dy, ddy):
        dz, ddz = self.sq2.backward(dy, ddy)
        da, dda = self.lin2.backward(dz, ddz)
        dz, ddz = self.sq1.backward(da, dda)
        dx, ddx = self.lin1.backward(dz, ddz)

        return dx, ddx

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
            loss = lambda x: x**2
            da, dda = 2 * a, np.array([[2]])
            dx, ddx = self.backward(da, dda)

            a_plus = self.forward(x_plus)
            a_minus = self.forward(x_minus)

            num_grad = (loss(a_plus) - loss(a_minus)) / (2 * self.eps)
            num_2grad = (loss(a_plus) - 2 * loss(a) + loss(a_minus)) / (self.eps ** 2)

            #self.assertTrue(abs(num_grad - dx[idx])<self.tol)
            #self.assertTrue(abs(num_2grad - ddx[idx]<np.sqrt(self.tol)))

    def test_grad_W(self):
        W1 = np.copy(self.lin1.W)
        for i in range(W1.shape[0]):
            for j in range(W1.shape[1]):

                mask = np.zeros_like(W1)
                mask[i,j]=1

                W1_plus = W1 + mask * self.eps
                W1_minus = W1 - mask * self.eps

                self.lin1.W[:]=W1
                a = self.forward(self.x)
                loss = lambda x: x ** 2
                da, dda = 2 * a, 2
                dx, ddx = self.backward(da, dda)

                dW1 = np.copy(self.lin1.d["W"])
                ddW1 = np.copy(self.lin1.dd["W"])

                self.lin1.W[:] = W1_plus
                a_plus = self.forward(self.x)

                self.lin1.W[:] = W1_minus
                a_minus = self.forward(self.x)

                num_grad = (loss(a_plus) - loss(a_minus)) / (2 * self.eps)
                num_2grad = (loss(a_plus) - 2 * loss(a) + loss(a_minus)) / (self.eps ** 2)

                self.assertTrue(abs(num_grad - dW1[i,j])<self.tol)
                self.assertTrue(abs(num_2grad - ddW1[i,j])<np.sqrt(self.tol))