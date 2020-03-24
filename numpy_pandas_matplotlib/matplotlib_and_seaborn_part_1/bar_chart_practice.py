# prerequisite package imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

pokemon = pd.read_csv('pokemon.csv')
print(pokemon.head())

# Task 1: There have been quite a few Pokémon introduced over the series' history. How many were introduced in each
# generation? Create a bar chart of these frequencies using the 'generation_id' column.
base_color = sb.color_palette()[0]
# n_pkmn_gen = pokemon.groupby(['generation_id'])['id'].agg(count=np.size)
# sb.barplot(n_pkmn_gen.index.values, n_pkmn_gen['count'], color=base_color)
sb.countplot(data=pokemon[['generation_id','id']], x='generation_id', color=base_color)
plt.show()

# Task 2: Each Pokémon species has one or two 'types' that play a part in its offensive and defensive capabilities.
# How frequent is each type? The code below creates a new dataframe that puts all of the type counts in a single column.
pkmn_types = pokemon.melt(id_vars=['id','species'],
                          value_vars=['type_1', 'type_2'],
                          var_name='type_level',
                          value_name='type').dropna()
# pkmn_types.head()
# Your task is to use this dataframe to create a relative frequency plot of the proportion of Pokémon with each type,
# sorted from most frequent to least. Hint: The sum across bars should be greater than 100%, since many Pokémon have
# two types. Keep this in mind when considering a denominator to compute relative frequencies.

type_counts = pkmn_types['type'].value_counts()
type_order = type_counts.index
n_pokemon = pokemon.shape[0]
max_type_count = type_counts[0]
max_prop = max_type_count / n_pokemon
ticks_prop = np.arange(0, max_prop, 0.02)
tick_names = ['{:0.2f}'.format(x) for x in ticks_prop]

sb.countplot(data=pkmn_types, y='type', color=base_color, order=type_order)
plt.xticks(ticks_prop * n_pokemon, tick_names)
plt.show()