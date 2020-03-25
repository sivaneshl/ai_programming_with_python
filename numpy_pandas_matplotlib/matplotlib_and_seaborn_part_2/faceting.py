import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

fuel_econ = pd.read_csv('fuel-econ.csv')

sedan_classes = ['Minicompact Cars', 'Subcompact Cars', 'Compact Cars', 'Midsize Cars', 'Large Cars']

bins = np.arange(fuel_econ['comb'].min(), fuel_econ['comb'].max()+2, 2)
g = sb.FacetGrid(data=fuel_econ, col='VClass', col_wrap=3, col_order=sedan_classes)
g.map(plt.hist, 'comb', bins=bins)
plt.show()