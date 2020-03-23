import numpy as np

x = np.random.random((3, 3))
print(x)

# random numbers with ints
# lower bound, upper bound, shape
x = np.random.randint(4, 15, (3, 2))
print(x)

# normal distribution with mean 0, std dev 0.1 of 1000X1000 matrix
x = np.random.normal(0, 0.1, (1000, 1000))
print(x)
print('mean', x.mean())
print('std', x.std())
print('max', x.max())
print('min', x.min())
print('# positive', (x > 0).sum())
print('# negative', (x < 0).sum())


