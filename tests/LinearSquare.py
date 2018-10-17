import numpy as np
import unittest

from LinearNode import LinearNode
from SquareNode import SquareNode

class LinearSquareTest(unittest.TestCase):


    def setUp(self):
        self.tol = 1e-6
        self.eps = 1e-5
        np.random.seed(0)

        W1 = np.random.rand(4, 3)
        b1 = np.random.rand(4, 1)
        W2 = np.random.rand(1, 4)
        b2 = np.random.rand(1, 1)

        batch_size = 1

        # first dimension is batch
        self.x = np.random.rand(batch_size, 3, 1)

        self.lin1 = LinearNode(W1, b1, batch_size)
        self.act1 = SquareNode(4, batch_size)
        self.lin2 = LinearNode(W2, b2, batch_size)
        self.act2 = SquareNode(1, batch_size)

    def forward(self, x):
        z = self.lin1.forward(x)
        a = self.act1.forward(z)
        z = self.lin2.forward(a)
        a = self.act2.forward(z)
        return a

    def backward_pass1(self, dy_dout):
        dy_din = self.act2.backward_pass1(dy_dout)
        dy_dout = dy_din
        dy_din = self.lin2.backward_pass1(dy_dout)
        dy_dout = dy_din
        dy_din = self.act1.backward_pass1(dy_dout)
        dy_dout = dy_din
        dy_din = self.lin1.backward_pass1(dy_dout)

        return dy_din

    def backward_pass2(self, ddy_ddout):
        ddy_ddin = self.act2.backward_pass2(ddy_ddout)
        ddy_ddout = ddy_ddin
        ddy_ddin = self.lin2.backward_pass2(ddy_ddout)
        ddy_ddout = ddy_ddin
        ddy_ddin = self.act1.backward_pass2(ddy_ddout)
        ddy_ddout = ddy_ddin
        ddy_ddin = self.lin1.backward_pass2(ddy_ddout)

        return ddy_ddin

    def test_passes(self):
        y = self.forward(self.x)
        dy_dx = self.backward_pass1(np.ones_like(y))
        ddy_ddx = self.backward_pass2(np.zeros_like(y))

    def test_grad_x(self):
        for idx in range(self.x.shape[1]):
            mask = np.zeros_like(self.x)
            mask[0, idx, 0]=1

            x_plus = self.x + mask * self.eps
            x_minus = self.x - mask * self.eps

            a = self.forward(self.x)
            loss = lambda a: a**2
            da, dda = 2*a, np.ones_like(a)*2
            dx= self.backward_pass1(da)
            ddx = self.backward_pass2(dda)

            a_plus = self.forward(x_plus)
            a_minus = self.forward(x_minus)

            num_grad = (loss(a_plus) - loss(a_minus)) / (2 * self.eps)
            num_2grad = (loss(a_plus) -2*loss(a) + loss(a_minus)) / (self.eps ** 2)

            diff_grad = abs(num_grad[0] - dx[0, 0, idx])
            rel_errror_grad = diff_grad / (abs(num_grad[0]) + abs(dx[0, 0, idx])) * 2

            diff_2grad = abs(num_2grad[0] - ddx[0, idx, idx])
            rel_errror_2grad = diff_2grad / (abs(num_2grad[0]) + abs(ddx[0,idx,idx])) * 2

            self.assertTrue(rel_errror_grad<self.tol)
            self.assertTrue(rel_errror_2grad<np.sqrt(self.tol))

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
                da, dda = 2 * a, 2
                dx, ddx = self.backward(da, dda)

                dW1 = np.copy(self.lin1.d["W"])
                ddW1 = np.copy(self.lin1.dd["W"])

                self.lin1.W[:] = W1_plus
                a_plus = self.forward(self.x)

                self.lin1.W[:] = W1_minus
                a_minus = self.forward(self.x)

                num_grad = (a_plus ** 2 - a_minus ** 2) / (2 * self.eps)
                num_2grad = (a_plus ** 2 - 2 * a ** 2 + a_minus ** 2) / (self.eps ** 2)

                self.assertTrue(abs(num_grad - dW1[i,j])<self.tol)
                self.assertTrue(abs(num_2grad - ddW1[i,j])<np.sqrt(self.tol))