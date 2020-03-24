import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

pokemon = pd.read_csv('pokemon.csv')

# Task 1: There are also variables in the dataset that don't have anything to do with the game mechanics, and are just
# there for flavor. Try plotting the distribution of Pokémon heights (given in meters). For this exercise, experiment
# with different axis limits as well as bin widths to see what gives the clearest view of the data.

bins = np.arange(0, pokemon['height'].max()+0.2, 0.2)
plt.hist(data=pokemon, x='height', bins=bins)
plt.xlim(0,6)
plt.show()


# Task 2: In this task, you should plot the distribution of Pokémon weights (given in kilograms). Due to the very large
# range of values taken, you will probably want to perform an axis transformation as part of your visualization workflow

print(np.log10(pokemon['weight'].describe()))
bins = 10 ** np.arange(-1, 3+0.1, 0.1)
plt.hist(data=pokemon, x='weight', bins=bins)
ticks = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
labels = ['{}'.format(x) for x in ticks]
plt.xscale('log')
plt.xticks(labels=labels, ticks=ticks)
plt.show()