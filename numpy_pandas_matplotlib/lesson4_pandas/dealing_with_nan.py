import pandas as pd

# We create a list of Python dictionaries
items2 = [{'bikes': 20, 'pants': 30, 'watches': 35, 'shirts': 15, 'shoes':8, 'suits':45},
          {'watches': 10, 'glasses': 50, 'bikes': 15, 'pants':5, 'shirts': 2, 'shoes':5, 'suits':7},
          {'bikes': 20, 'pants': 30, 'watches': 35, 'glasses': 4, 'shoes':10}]

# We create a DataFrame  and provide the row index
store_items = pd.DataFrame(items2, index = ['store 1', 'store 2', 'store 3'])

# We display the DataFrame
print(store_items)

# count null values
print(store_items.isnull())
print(store_items.isnull().sum())
print(store_items.isnull().sum().sum())
# count non null values
print(store_items.count())

# drop nan rows
print(store_items.dropna(axis=0))

# drop nan cols
print(store_items.dropna(axis=1))

# replace all nans with 0
print(store_items.fillna(0))

# replace all nans with values from prev rows or col
print(store_items.fillna(method='ffill', axis=0))
print(store_items.fillna(method='ffill', axis=1))

# replace all nans with values from next rows or col
print(store_items.fillna(method='backfill', axis=0))
print(store_items.fillna(method='backfill', axis=1))

# repalce all nans with linear interpolation
print(store_items.interpolate(method='linear', axis=0))
print(store_items.interpolate(method='linear', axis=1))

