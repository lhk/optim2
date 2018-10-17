import numpy as np
import unittest

from nodes.LinearNode import LinearNode
from nodes.SquareNode import SquareNode
from tests.NetworkTest import NetworkTest


class LinearSquareNetworkTest(NetworkTest, unittest.TestCase):

    def setUp(self):
        super().setUp()

        np.random.seed(0)

        W1 = np.random.rand(4, 3)
        b1 = np.random.rand(4, 1)
        W2 = np.random.rand(1, 4)
        b2 = np.random.rand(1, 1)

        batch_size = 1

        # first dimension is batch
        self.x = np.random.rand(batch_size, 3, 1)

        lin1 = LinearNode(W1, b1, batch_size)
        act1 = SquareNode(4, batch_size)
        lin2 = LinearNode(W2, b2, batch_size)
        act2 = SquareNode(1, batch_size)

        self.nodes = [lin1, act1, lin2, act2]
