import numpy as np

x = np.arange(1, 21).reshape(4, 5)
print(x)

y = x[1:, 2:]
print(y)

y = x[:3, 2:]
print(y)

y = x[:, 2]
print(y)
y = x[:, 2:3]
print(y)


z = x[1:, 2:]
z[2, 2] = 555
print(z)    # z is only a view of x
print(x)    # both x and z are updated

# to copy an array
x = np.arange(1, 21).reshape(4, 5)
print(x)
y = np.copy(x[1:, 2:])
print(y)
y[2, 2] = 555
print(y)
print(x)


# selecting
indices = np.array([1, 3])
print(indices)
y = x[indices, :]   # select 2nd and 4th row of x
print(y)
y = x[:, indices]   # select 2nd and 4th col of x
print(y)

# diagonal
print(x)
d = np.diag(x)
print(d)
d = np.diag(x, k=1) # one after the main diagonal
print(d)
d = np.diag(x, k=3) # 3 after the main diagonal
print(d)
d = np.diag(x, k=-1)    # one below the main diagonal
print(d)


# grab the unique elements
x = np.array([[1, 2, 3], [5, 2, 8], [1, 2, 3]])
print(x)
u = np.unique(x)
print(u)