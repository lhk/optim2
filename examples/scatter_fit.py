import numpy as np
import matplotlib.pyplot as plt

x = np.random.rand(2, 100)
y = np.zeros((1,100))
y[0, :50] = 1
x[0,:50] = x[0,:50]-0.2
x[0, 50:] = x[0,50:]+0.2

x = x - x.min()
x = x / x.max()

plt.scatter(x=x[0,:], y=x[1,:], c=y[0,:])
plt.show()

