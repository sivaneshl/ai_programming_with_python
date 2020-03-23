import numpy as np
import pandas as pd

fruits = pd.Series([10, 6, 3], ['apples', 'oranges', 'bananas'])
print(fruits)

fruits = fruits + 2
print(fruits)

fruits = fruits - 2
print(fruits)

fruits = fruits * 2
print(fruits)

fruits = fruits / 2
print(fruits)

fruits = np.sqrt(fruits)
print(fruits)

fruits = np.exp(fruits)
print(fruits)

fruits = np.power(fruits, 2)
print(fruits)

fruits = pd.Series([10, 6, 3], ['apples', 'oranges', 'bananas'])
print(fruits)

fruits['bananas'] += 2
fruits.iloc[0] -= 1
fruits[['apples', 'oranges']] *= 2
print(fruits)