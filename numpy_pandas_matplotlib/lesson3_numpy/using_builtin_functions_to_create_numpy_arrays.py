import numpy as np

my_list = [1, 2, 3, 4, 5]
print(np.array(my_list))

x = np.zeros((3, 4))
print(x)
print(x.dtype)

x = np.ones((3, 4))
print(x)

x = np.full((3, 4), 2)
print(x)
print(x.dtype)

# identity matrix
x = np.eye(5)
print(x)

# diagonal matrix
x = np.diag([10, 20, 30, 40, 50])
print(x)

# range
x = np.arange(2, 10, 2)
print(x)

# n evenly spaced numbers from start to stop (inclusive)
x = np.linspace(2, 10, 3)
print(x)

# n evenly spaced numbers from start to stop (exclusive)
x = np.linspace(2, 10, 3, endpoint=False)
print(x)

# convert a 1D array of 20 elements into a 4X5 2D array
x = np.arange(20)
print(x)
x = np.reshape(x, (4, 5))
print(x)
x = np.reshape(x, (10, 2))
print(x)

x = np.linspace(0, 50, 10, endpoint=False).reshape(5, 2)
print(x)
