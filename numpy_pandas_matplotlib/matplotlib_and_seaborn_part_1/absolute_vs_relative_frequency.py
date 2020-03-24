import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

pokemon = pd.read_csv('pokemon.csv')

pkmon_types = pokemon.melt(id_vars=['id', 'species'],
                           value_vars=['type_1', 'type_2'],
                           var_name='type_level',
                           value_name='type').dropna()

type_counts = pkmon_types['type'].value_counts()
type_order = type_counts.index

base_color = sb.color_palette()[0]
sb.countplot(data=pkmon_types, y='type', color=base_color, order=type_order)
# plt.show()

# compute the relative counts
n_pokemon = pokemon.shape[0]
max_type_count = type_counts[0]
max_prop = max_type_count / n_pokemon
print(max_prop)

tick_props = np.arange(0, max_prop, 0.02)
print(tick_props)

tick_names = ['{:0.2f}'.format(v) for v in tick_props]
print(tick_names)

sb.countplot(data=pkmon_types, y='type', color=base_color, order=type_order)
plt.xticks(tick_props * n_pokemon, tick_names)
plt.xlabel('proportion')

for i in range(type_counts.shape[0]):
    count = type_counts[i]
    pct_string = '{:0.1f}%'.format(100*count/n_pokemon)
    plt.text(count + 1, i, pct_string, va='center')

plt.show()
