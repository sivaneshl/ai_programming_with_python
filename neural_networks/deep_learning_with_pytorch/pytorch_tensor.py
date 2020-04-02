import numpy as np
import torch

import torchvision

# create a random tensor
x = torch.rand(3, 2)
print(x)

y = torch.ones(x.size())
print(y)

z = x + y
print(z)

# slicing tensors
print(z[0]) # first row
print(z[:, 1:]) # all rows second column

# 2 types of methods
print(z.add(1)) # add 1 to z and creates a new tensor
print(z)    # z is unchanged
print(z.add_(1))    # adds 1 to z but inplace
print(z)    # z is changed

# reshaping
print(z.size())
z.resize_(2, 3)
print(z)

# numpy to torch and back
a = np.random.rand(4, 3)
print(a)
b = torch.from_numpy(a)
print(b)
# back to numpy
c = b.numpy()
print(c)

# numpy array and tensor share memory - so changes made to tensor will also affect numpy array
b.mul_(2)
print(b)
print(a)