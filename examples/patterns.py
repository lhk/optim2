import numpy as np

def circles():
    # set up the data
    # circle one
    num_samples = 100
    phi = np.linspace(0, np.pi * 5, num_samples)
    r = np.linspace(0, 10, num_samples)

    x1 = np.cos(phi) * r
    y1 = np.sin(phi) * r

    labels1 = np.zeros((num_samples,))

    # circle two
    num_samples = 100
    phi = np.linspace(0 + np.pi / 2, np.pi * 5 + np.pi / 1.3, num_samples)
    r = np.linspace(0, 10, num_samples)

    x2 = np.cos(phi) * r
    y2 = np.sin(phi) * r

    labels2 = np.ones((num_samples,))

    x = np.concatenate([x1, x2]).reshape((-1, 1))
    y = np.concatenate([y1, y2]).reshape((-1, 1))
    labels = np.concatenate([labels1, labels2])

    return x, y, labels

def sides():
    num_samples = 100
    x1 =np.random.rand(num_samples)*0.7
    x2 = np.random.rand(num_samples)*0.7 + 0.3
    y1 = np.random.rand(num_samples)
    y2 = np.random.rand(num_samples)

    labels1 = np.zeros((num_samples,))
    labels2 = np.ones((num_samples,))

    x = np.concatenate([x1, x2]).reshape((-1, 1))
    y = np.concatenate([y1, y2]).reshape((-1, 1))
    labels = np.concatenate([labels1, labels2])

    return x, y, labels