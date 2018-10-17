import numpy as np
import unittest

from LinearNode import LinearNode
from TanhNode import TanhNode

class NetworkTest(unittest.TestCase):

    def setUp(self):
        self.tol = 1e-6
        self.eps = 1e-3
        np.random.seed(0)

        self.nodes=[]

    def forward(self, x):
        for node in self.nodes:
            x = node.forward(x)
        return x

    def backward_pass1(self, dy_dout):
        for node in reversed(self.nodes):
            dy_dout = node.backward_pass1(dy_dout)
        return dy_dout

    def backward_pass2(self, ddy_ddout):
        for node in reversed(self.nodes):
            dy_dout = node.backward_pass2(ddy_ddout)
        return ddy_ddout

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
            rel_errror_grad = diff_grad / (abs(num_grad[0]) + abs(dx[0, 0, idx]))

            diff_2grad = abs(num_2grad[0] - ddx[0, idx, idx])
            rel_errror_2grad = diff_2grad / (abs(num_2grad[0]) + abs(ddx[0,idx,idx]))

            self.assertTrue(rel_errror_grad<self.tol)
            self.assertTrue(rel_errror_2grad<np.sqrt(self.tol))

    def test_grad_W(self):
        for node in self.nodes:
            if node.param_ids==[]:
                continue
            for param_id in node.param_ids:
                param_value = np.copy(node.params[param_id])
                for idx in np.ndindex(param_value.shape):
                    mask = np.zeros_like(param_value)
                    mask[idx]=1

                    param_plus = param_value + mask * self.eps
                    param_minus = param_value - mask * self.eps

                    node.params[param_id][:] = param_value
                    a = self.forward(self.x)
                    loss = lambda a: a**2
                    da, dda = 2 * a, np.ones_like(a)*2
                    dx= self.backward_pass1(da)
                    ddx = self.backward_pass2(dda)

                    dparam = np.copy(node.J_y[param_id])
                    ddparam = np.copy(node.H_y[param_id])

                    node.params[param_id][:] = param_plus
                    a_plus = self.forward(self.x)

                    node.params[param_id][:] = param_minus
                    a_minus = self.forward(self.x)

                    num_grad = (loss(a_plus) - loss(a_minus)) / (2 * self.eps)
                    num_2grad = (loss(a_plus) - 2 * loss(a) + loss(a_minus)) / (self.eps ** 2)

                    diff_grad = abs(num_grad[0] - dparam[(0, *idx)])
                    rel_errror_grad = diff_grad / (abs(num_grad[0]) + abs(dparam[(0, *idx)]))

                    diff_2grad = abs(num_2grad[0] - ddparam[(0, *idx, *idx)])
                    rel_errror_2grad = diff_2grad / (abs(num_2grad[0]) + abs(ddparam[(0, *idx, *idx)]))

                    self.assertTrue(rel_errror_grad < self.tol)
                    self.assertTrue(rel_errror_2grad < np.sqrt(self.tol))