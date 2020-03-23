import pandas as pd

groceries = pd.Series(data=[30, 6, 'Yes', 'No'], index=['eggs', 'apple', 'milk', 'bread'])
print(groceries)

print(groceries.shape)
print(groceries.ndim)
print(groceries.size)

# print the index labels
print(groceries.index)

# print the values
print(groceries.values)

# check if a label is in the index
print('banana' in groceries)
print('banana' in groceries.index)

