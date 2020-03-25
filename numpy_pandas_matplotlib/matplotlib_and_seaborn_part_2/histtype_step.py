import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

fuel_econ = pd.read_csv('fuel-econ.csv')

sedan_classes = ['Minicompact Cars', 'Subcompact Cars', 'Compact Cars', 'Midsize Cars', 'Large Cars']
vclasses = pd.api.types.CategoricalDtype(ordered=True, categories=sedan_classes)
fuel_econ['VClass'] = fuel_econ['VClass'].astype(vclasses)

bins = np.arange(fuel_econ['comb'].min(), fuel_econ['comb'].max()+2, 2)
g = sb.FacetGrid(data=fuel_econ, hue='VClass', size=5)
g.map(plt.hist, 'comb', bins=bins, histtype='step')
g.add_legend()
plt.show()

