import numpy as np

x = np.random.randint(1, 11, size=(10,))
print(x)

# using sort as a funciton leaves the original array unchanged
print(np.sort(x))
print(x)

# using sort as a method changes the original array
x.sort()
print(x)

# 2d arrays
x = np.random.randint(1, 11, size=(5, 5))
print(x)
print(np.sort(x, axis=0))   # sort by rows
print(np.sort(x, axis=1))   # sort by cols
