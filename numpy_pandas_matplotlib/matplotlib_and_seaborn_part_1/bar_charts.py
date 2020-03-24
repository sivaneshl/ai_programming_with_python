import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

pokemon = pd.read_csv('pokemon.csv')
print(pokemon.shape)
print(pokemon.head())

base_color = sb.color_palette()[0]
gen_order = pokemon['generation_id'].value_counts().index
sb.countplot(data=pokemon, x='generation_id', color=base_color, order=gen_order)
plt.show()

type_1_order = pokemon['type_1'].value_counts().index
sb.countplot(data=pokemon, x='type_1', color=base_color, order=type_1_order)
plt.xticks(rotation=90)
plt.show()

type_2_order = pokemon['type_2'].value_counts().index
sb.countplot(data=pokemon, y='type_2', color=base_color, order=type_2_order)
plt.show()