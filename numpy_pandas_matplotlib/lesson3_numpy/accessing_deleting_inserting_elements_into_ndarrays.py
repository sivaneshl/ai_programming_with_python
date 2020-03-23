import numpy as np

# accessing 1d array elements
x = np.array([1, 2, 3, 4, 5])
print('1st element', x[0])
print('2nd element', x[1])
print('5th element', x[4])

# modifying 1d array elements
x[3] = 20
print(x)

# accessing 2d array elaments
x = np.arange(1, 10).reshape(3, 3)
print(x)
print(x[0, 2])
print(x[2, 0])

# modifying 2d array elements
x[2, 1] = 20
print(x)

# deleting 1d array elements
x = np.array([1, 2, 3, 4, 5])
x = np.delete(x, [0, 4])   # delete the first and last element of x
print(x)

# deleting 2d array elements
y = np.arange(1, 10).reshape(3, 3)
print(y)
w = np.delete(y, 0, axis=0)     # delete 1st row of y
print(w)
v = np.delete(y, [0, 2], axis=1)    # delete first and last col of y
print(v)

# append an element to an 1d array
x = np.array([1, 2, 3, 4, 5])
print(x)
x = np.append(x, 6)
print(x)
x = np.append(x, [7, 8])
print(x)

# append new rows and cols to a 2d array
x = np.arange(1, 10).reshape(3, 3)
w = np.append(x, [[10, 11, 12]], axis=0)    # append new row
print(w)
v = np.append(x, [[10], [11], [12]], axis=1)    # append new col
print(v)

# insert values to 1d arrays
x = np.array([1, 2, 5, 6, 7])
x = np.insert(x, 2, [3, 4])
print(x)

# insert values to 2d arrays
x = np.array([[1, 2, 3], [7, 8, 9]])
print(x)
y = np.insert(x, 1, [4, 5, 6], axis=0)  # insert row
print(y)
z = np.insert(x, 2, [4, 5], axis=1)     # insert col
print(z)



