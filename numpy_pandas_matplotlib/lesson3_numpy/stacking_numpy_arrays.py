import numpy as np

x = np.array([1, 2])
print(x)
y = np.array([[3, 4], [5, 6]])
print(y)

# stack  x on top of y
z = np.vstack((x, y))
print(z)

# stack x and y side by side
z = np.hstack((x.reshape(2, 1), y))
print(z)