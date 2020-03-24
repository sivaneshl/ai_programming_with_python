# prerequisite package imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

pokemon = pd.read_csv('pokemon.csv')

# Task: Pokémon have a number of different statistics that describe their combat capabilities. Here, create a histogram
# that depicts the distribution of 'special-defense' values taken.
# Hint: Try playing around with different bin width sizes to see what best depicts the data.
bins = np.arange(0, pokemon['special-defense'].max()+5, 5)
plt.hist(data=pokemon, x='special-defense', bins=bins)
plt.show()