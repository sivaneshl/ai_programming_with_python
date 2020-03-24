import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

pokemon = pd.read_csv('pokemon.csv')

print(np.log10(pokemon['weight'].describe()))   # min is -1 and max is 3 --> this gives us the bin edges

plt.figure(figsize=[10, 5])

plt.subplot(1, 2, 1)
bins = np.arange(0, pokemon['weight'].max()+40, 40)
plt.hist(data=pokemon, x='weight', bins=bins)

plt.subplot(1, 2,  2)
bins = 10 ** np.arange(-1, 3+0.1, 0.1)
ticks = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
labels = ['{}'.format(x) for x in ticks]
plt.hist(data=pokemon, x='weight', bins=bins)
plt.xscale('log')
plt.xticks(ticks, labels)
plt.show()


# square root transformation
def sqrt_trans(x, inverse=False):
    if not inverse:
        return np.sqrt(x)
    else:
        return x**2

    
bins = np.arange(0, sqrt_trans(pokemon['weight'].max())+1, 1)
plt.hist(data=pokemon, x='weight', bins=bins)
ticks = np.arange(0, sqrt_trans(pokemon['weight'].max())+10, 10)
labels = sqrt_trans(ticks, inverse=True).astype(int)
plt.xticks(ticks, labels)
plt.show()