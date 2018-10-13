import numpy as np
import matplotlib.pyplot as plt

# circle one
num_samples=100
phi = np.linspace(0, np.pi * 5, num_samples)
r = np.linspace(0, 10, num_samples)

x1 = np.cos(phi)*r
y1 = np.sin(phi)*r

labels1 = np.zeros((num_samples,))

# circle two
num_samples=100
phi = np.linspace(0 + np.pi/2, np.pi * 5 + np.pi/1.3, num_samples)
r = np.linspace(0, 10, num_samples)

x2 = np.cos(phi)*r
y2 = np.sin(phi)*r

labels2 = np.ones((num_samples,))

x = np.concatenate([x1, x2])
y = np.concatenate([y1, y2])
labels = np.concatenate([labels1, labels2])

plt.scatter(x, y, c=labels)
plt.show()

