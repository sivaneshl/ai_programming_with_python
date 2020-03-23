import pandas as pd

groceries = pd.Series(data=[30, 6, 'Yes', 'No'], index=['eggs', 'apple', 'milk', 'bread'])
print(groceries)

# access elements with their index labels
print(groceries['eggs'])
print(groceries[['apple', 'bread']])

# access elements with their numeric index
print(groceries[0])
print(groceries[[1, 3]])
print(groceries[-1])

print(groceries.loc[['apple', 'eggs']])
print(groceries.iloc[[0, 2]])

# update an element
groceries['eggs'] = 2
print(groceries)

# delete an element
groceries.drop('apple', inplace=True)
print(groceries)