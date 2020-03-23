# To Do:
# You will start by importing NumPy and creating a rank 2 ndarray of random integers between 0 and 5,000 (inclusive)
# with 1000 rows and 20 columns. This array will simulate a dataset with a wide range of values. Fill in the code below

import numpy as np

# Create a 1000 x 20 ndarray with random integers in the half-open interval [0, 5001).
X = np.random.randint(0, 5001, (1000, 20))

# print the shape of X
print(X.shape)

# Now that you created the array we will mean normalize it. We will perform mean normalization using the following equation:
#
# Norm_Col𝑖=(Col𝑖−𝜇𝑖)/𝜎𝑖
# where  Col𝑖  is the  𝑖 th column of  𝑋 ,  𝜇𝑖  is average of the values in the  𝑖 th column of  𝑋 , and  𝜎𝑖  is the
# standard deviation of the values in the  𝑖 th column of  𝑋 . In other words, mean normalization is performed by
# subtracting from each column of  𝑋  the average of its values, and then by dividing by the standard deviation of its
# values. In the space below, you will first calculate the average and standard deviation of each column of  𝑋 .

# Average of the values in each column of X
ave_cols = np.mean(X, axis=0)

# Standard Deviation of the values in each column of X
std_cols = np.std(X, axis=0)

# If you have done the above calculations correctly, then ave_cols and std_cols, should both be vectors with shape
# (20,) since  𝑋  has 20 columns. You can verify this by filling the code below:

# Print the shape of ave_cols
print(ave_cols.shape)

# Print the shape of std_cols
print(std_cols.shape)

# You can now take advantage of Broadcasting to calculate the mean normalized version of  𝑋  in just one line of code
# using the equation above. Fill in the code below

# Mean normalize X
X_norm = (X - ave_cols) / std_cols

# If you have performed the mean normalization correctly, then the average of all the elements in  𝑋norm  should be
# close to zero, and they should be evenly distributed in some small interval around zero. You can verify this by
# filing the code below:

# Print the average of all the values of X_norm
print(np.mean(X_norm))

# Print the average of the minimum value in each column of X_norm
print(np.min(X_norm))

# Print the average of the maximum value in each column of X_norm
print(np.max(X_norm))

# You should note that since  𝑋  was created using random integers, the above values will vary.

# In the space below create a rank 1 ndarray that contains a random permutation of the row indices of X_norm.
# You can do this in one line of code by extracting the number of rows of X_norm using the shape attribute and then
# passing it to the np.random.permutation() function. Remember the shape attribute returns a tuple with two numbers in
# the form (rows,columns).

# Create a rank 1 ndarray that contains a random permutation of the row indices of `X_norm`
row_indices = np.random.permutation(X_norm.shape[0])

# Now you can create the three datasets using the row_indices ndarray to select the rows that will go into each dataset.
# Rememeber that the Training Set contains 60% of the data, the Cross Validation Set contains 20% of the data, and the
# Test Set contains 20% of the data. Each set requires just one line of code to create. Fill in the code below

# Make any necessary calculations.
# You can save your calculations into variables to use later.
#
#
# Create a Training Set
X_train = X_norm[row_indices[:600], :]

# Create a Cross Validation Set
X_crossVal = X_norm[row_indices[600:800], :]

# Create a Test Set
X_test = X_norm[row_indices[800:], :]

# If you performed the above calculations correctly, then X_tain should have 600 rows and 20 columns, X_crossVal should
# have 200 rows and 20 columns, and X_test should have 200 rows and 20 columns. You can verify this by filling the code
# below:

# Print the shape of X_train
print(X_train.shape)

# Print the shape of X_crossVal
print(X_crossVal.shape)

# Print the shape of X_test
print(X_test.shape)
