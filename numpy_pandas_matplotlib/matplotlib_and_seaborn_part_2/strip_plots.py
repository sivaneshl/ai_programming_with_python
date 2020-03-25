import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

fuel_econ = pd.read_csv('fuel-econ.csv')

sedan_classes = ['Minicompact Cars', 'Subcompact Cars', 'Compact Cars', 'Midsize Cars', 'Large Cars']
vclasses = pd.api.types.CategoricalDtype(ordered=True, categories=sedan_classes)
fuel_econ['VClass'] = fuel_econ['VClass'].astype(vclasses)

plt.figure(figsize=[10, 5])
base_color = sb.color_palette()[0]

# strip plot
plt.subplot(1, 2, 1)
ax1 = sb.stripplot(data=fuel_econ, x='comb', y='VClass', color=base_color)

# violin plot with inner strips
plt.subplot(1, 2, 2)
sb.violinplot(data=fuel_econ, x='comb', y='VClass', color=base_color, inner='stick')
plt.ylim = ax1.get_ylim()
plt.show()