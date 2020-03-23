import numpy as np

x = np.array([1, 2, 3, 4])
y = np.array([5, 6, 7, 8])
# element wise operations
print(x + y)
print(np.add(x, y))

print(x - y)
print(np.subtract(x, y))

print(x * y)
print(np.multiply(x, y))

print(x / y)
print(np.divide(x, y))


# on 2d arrays
x = np.array([1, 2, 3, 4]).reshape(2, 2)
y = np.array([5, 6, 7, 8]).reshape(2, 2)
print(x + y)
print(np.add(x, y))


# sqrt
x = np.array([1, 2, 3, 4])
print(np.sqrt(x))

# exp
print(np.exp(x))

# power
print(np.power(x, 2))

# average of matrix
x = np.array([1, 2, 3, 4]).reshape(2, 2)
print('mean', np.mean(x))
print('row mean', np.mean(x, axis=0))
print('col mean', np.mean(x, axis=1))

print(np.sum(x))
print(np.std(x))
print(np.median(x))
print(np.max(x))
print(np.min(x))


# add a number to all elements of an array
x = np.array([1, 2, 3, 4]).reshape(2, 2)
print(x + 3)
print(x - 3)
print(x * 3)
print(x / 3)

# add 2 arrays of different shapes
x = np.array([0, 1, 2])
y = np.arange(0, 9).reshape(3, 3)
print(x)
print(y)
print(y+x)

x = np.arange(3).reshape(3, 1)
y = np.arange(0, 9).reshape(3, 3)
print(x)
print(y)
print(y+x)




