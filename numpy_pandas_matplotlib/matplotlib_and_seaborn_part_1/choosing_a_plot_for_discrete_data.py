import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

die_rolls = np.random.randint(1, 13, 100, dtype=int)

plt.figure(figsize=[10, 5])

# histogram on the left, bin edges on integers
plt.subplot(1, 2, 1)
bin_edges = np.arange(2, 12+1.1, 1)
plt.hist(die_rolls, bins=bin_edges)
plt.xticks(np.arange(2, 12+1, 1))
plt.subplot(1, 2, 2)
bin_edges = np.arange(1.5, 12.5+1, 1)
plt.hist(die_rolls, bins=bin_edges)
plt.xticks(np.arange(2, 12+1, 1))
plt.show()

# rwidth
bin_edges = np.arange(1.5, 12.5+1, 1)
plt.hist(die_rolls, bins=bin_edges, rwidth=0.7)
plt.xticks(np.arange(2, 12+1, 1))
plt.show()