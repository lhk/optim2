import numpy as np

class Node:
    """
    this is just a template
    """
    def __init__(self):
        self.inputs=[]
        self.outputs=[]

    def forward(self, x):
        self.x= x

    def dx(self, dy):
        self.dy=dy


    def ddx(self, dy):
        self.dy = dy


