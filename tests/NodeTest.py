import unittest
from nodes.Node import Node
import numpy as np

class NodeTest(unittest.TestCase):

    def setUp(self):
        self.node = Node(1, 1, 1)

    def test_forward(self):
        self.assertRaises(NotImplementedError, self.node.forward)

    def test_update_J_ax(self):
        self.assertRaises(NotImplementedError, self.node.update_J_ax)

    def test_update_H_ax(self):
        self.assertRaises(NotImplementedError, self.node.update_H_ax)
