import pandas as pd

items = {'Bob' : pd.Series(data = [245, 25, 55], index = ['bike', 'pants', 'watch']),
         'Alice' : pd.Series(data = [40, 110, 500, 45], index = ['book', 'glasses', 'bike', 'pants'])}

print(type(items))

shopping_carts = pd.DataFrame(items)
print(shopping_carts)

data = {'Bob' : pd.Series(data = [245, 25, 55]), 'Alice' : pd.Series(data = [40, 110, 500, 45])}
df = pd.DataFrame(data)
print(df)

print(shopping_carts.index)
print(shopping_carts.columns)
print(shopping_carts.values)
print(shopping_carts.shape)
print(shopping_carts.ndim)
print(shopping_carts.size)

bob_shopping_cart = pd.DataFrame(items, columns=['Bob'])
print(bob_shopping_cart)

sel_shopping_cart = pd.DataFrame(items, index=['pants', 'book'])
print(sel_shopping_cart)

alice_sel_shopping_cart = pd.DataFrame(items, columns=['Alice'], index=['glasses', 'bike'])
print(alice_sel_shopping_cart)

# from dict
data = {'Integers': [1, 2, 3],
        'Floats': [4.5, 8.2, 9.6]}
df = pd.DataFrame(data, index=['label1', 'label2', 'label3'])
print(df)

# from a list of dict
items = [{'bikes': 20, 'pants': 30, 'watches': 35},
         {'watches': 10, 'glasses': 50, 'bikes': 15,  'pants': 5}]
df = pd.DataFrame(items, index=['store1', 'store2'])
print(df)

