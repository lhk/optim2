import numpy as np
import unittest

from LinearNode import LinearNode
from TanhNode import TanhNode

class LinearTanhTest(unittest.TestCase):


    def setUp(self):
        self.tol = 1e-6
        self.eps = 1e-5
        np.random.seed(0)

        W1 = np.random.rand(4, 3)
        b1 = np.random.rand(4, 1)
        W2 = np.random.rand(1, 4)
        b2 = np.random.rand(1, 1)

        self.x = np.random.rand(3, 1)

        self.lin1 = LinearNode(W1, b1)
        self.tan1 = TanhNode()
        self.lin2 = LinearNode(W2, b2)
        self.tan2 = TanhNode()

    def forward_pass1(self, x):
        z = self.lin1.forward_pass1(x)
        a = self.tan1.forward_pass1(z)
        z = self.lin2.forward_pass1(a)
        a = self.tan2.forward_pass1(z)
        return a

    def backward_pass1(self, dy_dout):
        dy_din = self.tan2.backward_pass1(dy_dout)
        dy_dout = dy_din
        dy_din = self.lin2.backward_pass1(dy_dout)
        dy_dout = dy_din
        dy_din = self.tan1.backward_pass1(dy_dout)
        dy_dout = dy_din
        dy_din = self.lin1.backward_pass1(dy_dout)

        return dy_din

    def forward_pass2(self, x):
        dx_dx = np.ones_like(x)
        dout_dx = self.lin1.forward_pass2(dx_dx)
        din_dx = dout_dx
        dout_dx = self.tan1.forward_pass2(din_dx)
        din_dx = dout_dx
        dout_dx = self.lin2.forward_pass2(din_dx)
        din_dx = dout_dx
        dout_dx = self.tan2.forward_pass2(din_dx)

        return dout_dx

    def backward_pass2(self, ddy_ddout):
        ddy_ddin = self.tan2.backward_pass2(ddy_ddout)
        ddy_ddout = ddy_ddin
        ddy_ddin = self.lin2.backward_pass2(ddy_ddout)
        ddy_ddout = ddy_ddin
        ddy_ddin = self.tan1.backward_pass2(ddy_ddout)
        ddy_ddout = ddy_ddin
        ddy_ddin = self.lin1.backward_pass2(ddy_ddout)

        return ddy_ddin

    def test_passes(self):
        y = self.forward_pass1(self.x)
        dy_dx = self.backward_pass1(np.ones_like(y))
        ddy_ddx = self.backward_pass2(np.zeros_like(y))

    def test_grad_x(self):
        for idx in range(self.x.shape[0]):
            mask = np.zeros_like(self.x)
            mask[idx]=1

            x_plus = self.x + mask * self.eps
            x_minus = self.x - mask * self.eps

            a = self.forward_pass1(self.x)
            loss = lambda a: a
            da, dda = 1, 0
            dx= self.backward_pass1(da)
            ddx = self.backward_pass2(dda)

            a_plus = self.forward_pass1(x_plus)
            a_minus = self.forward_pass1(x_minus)

            num_grad = (loss(a_plus) - loss(a_minus)) / (2 * self.eps)
            num_2grad = (loss(a_plus) -2*loss(a) + loss(a_minus)) / (self.eps ** 2)

            self.assertTrue(abs(num_grad - dx[idx])<self.tol)
            self.assertTrue(abs(num_2grad - ddx[idx]<np.sqrt(self.tol)))

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