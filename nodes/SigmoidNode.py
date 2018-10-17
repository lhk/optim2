import numpy as np
from nodes.Node import Node

class SigmoidNode(Node):
    def __init__(self, n, batch_size):

        # sigmoid maps n values to n values
        super().__init__(n,n,batch_size)

        #no params
        self.param_ids=[]

    def forward(self, x):

        self.x= x
        self.exp_negx = np.exp(-x)
        self.a = 1/(1+self.exp_negx)
        return self.a

    def update_J_ax(self):
        n = self.n
        batch_size = self.batch_size
        self.J_ax = np.zeros((batch_size, n,n))
        self.J_ax[:, np.arange(n), np.arange(n)] = (self.exp_negx/(self.exp_negx +1)**2).reshape((batch_size, n))

    def update_H_ax(self):
        n = self.n
        batch_size = self.batch_size
        self.H_ax = np.zeros((batch_size, n, n, n))
        self.H_ax[:, np.arange(n), np.arange(self.n), np.arange(self.n)] =  (((2*np.exp(-2*self.x)/(self.exp_negx+1)**3) - self.exp_negx/(self.exp_negx +1)**2)).reshape((batch_size, n))