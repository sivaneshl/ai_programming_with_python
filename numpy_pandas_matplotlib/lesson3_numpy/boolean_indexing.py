import numpy as np

x = np.arange(1, 26).reshape(5, 5)
print(x)

# elements that are greater than 10
print(x[x>10])

# elements that are greater than 7 and less than 17
print(x[(x > 7) & (x < 17)])

# assign these elements to -1
x[(x > 7) & (x < 17)] = -1
print(x)

