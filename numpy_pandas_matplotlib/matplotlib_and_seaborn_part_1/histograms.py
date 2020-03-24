import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

pokemon = pd.read_csv('pokemon.csv')

plt.hist(data=pokemon, x='speed')   # default bins 10
plt.show()
plt.hist(data=pokemon, x='speed', bins=20)  # change bin size
plt.show()

bins = np.arange(0, pokemon['speed'].max()+5, 5)
plt.hist(data=pokemon, x='speed', bins=bins)
plt.show()

# using seaborn
sb.distplot(pokemon['speed'])
plt.show()
sb.distplot(pokemon['speed'], kde=False)    # turn off the density line
plt.show()

sb.distplot(pokemon['speed'], bins=bins)
plt.show()
sb.distplot(pokemon['speed'], bins=bins, hist_kws={'alpha': 1}, kde=False)
plt.show()
