import numpy as np

x = np.array([1,2,3,4,5])
print(x)
print(type(x))
print(x.dtype)
print(x.shape)

# 2d array
y = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(y)
print(y.shape)
print(y.size)

z = np.array(['hello', 'world'])
print(z)
print(type(z))
print(z.dtype)
print(z.shape)

z1 = np.array([1, 2, 'hello'])
print(z)
print(type(z1))
print(z1.dtype)
print(z1.shape)

z2 = np.array([1, 2, 2.5, 3])
print(z2)
print(type(z2))
print(z2.dtype)
print(z2.shape)

z3 = np.array([1.5, 2, 2.5, 3], dtype=np.int64)
print(z3)
print(type(z3))
print(z3.dtype)
print(z3.shape)

