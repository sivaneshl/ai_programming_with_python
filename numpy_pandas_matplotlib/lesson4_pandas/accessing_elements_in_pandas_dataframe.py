import pandas as pd

items = [{'bikes': 20, 'pants': 30, 'watches': 35},
         {'watches': 10, 'glasses': 50, 'bikes': 15,  'pants': 5}]
store_items = pd.DataFrame(items, index=['store1', 'store2'])
print(store_items)

# access by cols
print(store_items[['bikes']])
print(store_items[['bikes', 'pants']])

# access by row
print(store_items.loc['store1'])

# access by col and row
print(store_items['bikes']['store2'])

# adding new cols
store_items['shirts'] = [5, 2]
print(store_items)
store_items['suits'] = store_items['shirts'] + store_items['pants']
print(store_items)

# adding new rows
store3_items = [{'bikes': 20, 'pants': 30, 'watches': 35, 'glasses': 4}]
new_store = pd.DataFrame(store3_items, index=['store3'])
print(new_store)
store_items = store_items.append(new_store)
print(store_items)

# adding new cols for only specific rows
store_items['new_watches'] = store_items['watches'][1:]
print(store_items)

# insert column
store_items.insert(5, 'shoes', [8, 5, 10])
print(store_items)

# pop allows to delete cols
store_items.pop('new_watches')
print(store_items)

# drop allows to delete rows and cols
store_items = store_items.drop(['shoes', 'watches'], axis=1)
print(store_items)
store_items = store_items.drop('store3', axis=0)
print(store_items)

# rename col labels
store_items = store_items.rename(columns={'bikes': 'hats'})
print(store_items)

# rename row labels
store_items = store_items.rename(index={'store2': 'laststore'})
print(store_items)


