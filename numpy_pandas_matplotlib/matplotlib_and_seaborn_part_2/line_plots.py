import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

fuel_econ = pd.read_csv('fuel-econ.csv')

bin_edges = np.arange(fuel_econ['displ'].min(), fuel_econ['displ'].max()+0.2, 0.2)
bin_centers = bin_edges[:-1] + 0.1
displ_binned = pd.cut(fuel_econ['displ'], bin_edges, include_lowest=True)
comb_mean = fuel_econ['comb'].groupby(displ_binned).mean()
comb_std = fuel_econ['comb'].groupby(displ_binned).std()

plt.errorbar(x=bin_centers, y=comb_mean, yerr=comb_std)
plt.xlabel('Displacement (l)')
plt.ylabel('Avg. Combined Fuel Eff. (mpg)')
plt.show()
