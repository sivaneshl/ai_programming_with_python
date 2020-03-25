import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

fuel_econ = pd.read_csv('fuel-econ.csv')

sedan_classes = ['Minicompact Cars', 'Subcompact Cars', 'Compact Cars', 'Midsize Cars', 'Large Cars']
vclasses = pd.api.types.CategoricalDtype(ordered=True, categories=sedan_classes)
fuel_econ['VClass'] = fuel_econ['VClass'].astype(vclasses)

plt.figure(figsize = [12, 5])
base_color = sb.color_palette()[0]

plt.subplot(1, 3, 1)
ax1 = sb.violinplot(data=fuel_econ, x='VClass', y='displ', color=base_color)
plt.xticks(rotation=90)

plt.subplot(1, 3, 2)
sb.boxplot(data=fuel_econ, x='VClass', y='displ', color=base_color)
plt.ylim = ax1.get_ylim()
plt.xticks(rotation=90)

plt.subplot(1, 3, 3)
sb.swarmplot(data=fuel_econ, x='VClass', y='displ', color=base_color)
plt.ylim = ax1.get_ylim()
plt.xticks(rotation=90)

plt.show()


