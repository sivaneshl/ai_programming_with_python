import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

fuel_econ = pd.read_csv('fuel-econ.csv')

plt.hist2d(data=fuel_econ, x='displ', y='comb', cmin=0.5)
plt.colorbar()
plt.xlabel('Displacement (l)')
plt.ylabel('Combined Fuel Eff. (mpg)')
plt.show()

print(fuel_econ[['displ', 'comb']].describe())
bins_x = np.arange(0.6, 7+0.3, 0.3)
bins_y = np.arange(12, 58+3, 3)
plt.hist2d(data=fuel_econ, x='displ', y='comb', cmin=0.5,  bins=[bins_x, bins_y])
plt.colorbar()
plt.xlabel('Displacement (l)')
plt.ylabel('Combined Fuel Eff. (mpg)')
plt.show()

# hist2d returns a number of different variables, including an array of counts
h2d = plt.hist2d(data=fuel_econ, x='displ', y='comb', cmin=0.5,  bins=[bins_x, bins_y])
counts = h2d[0]
# loop through the cell counts and add text annotations for each
for i in range(counts.shape[0]):
    for j in range(counts.shape[1]):
        c = counts[i, j]
        if c >= 100:
            plt.text(bins_x[i]+0.15, bins_y[j]+1.5, int(c),
                     ha='center', va='center', color='black')
        elif c > 0:
            plt.text(bins_x[i]+0.15, bins_y[j]+1.5, int(c),
                     ha='center', va='center', color='white')
plt.colorbar()
plt.xlabel('Displacement (l)')
plt.ylabel('Combined Fuel Eff. (mpg)')
plt.show()