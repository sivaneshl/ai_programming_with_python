import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

fuel_econ = pd.read_csv('fuel-econ.csv')

sedan_classes = ['Minicompact Cars', 'Subcompact Cars', 'Compact Cars', 'Midsize Cars', 'Large Cars']
vclasses = pd.api.types.CategoricalDtype(ordered=True, categories=sedan_classes)
fuel_econ['VClass'] = fuel_econ['VClass'].astype(vclasses)


base_color = sb.color_palette()[0]
sb.barplot(data=fuel_econ, x='VClass', y='comb', color=base_color)
plt.xticks(rotation=15)
plt.ylabel('Avg. Combined Fuel Eff. (mpg)')
plt.show()

# without the mean line
sb.barplot(data=fuel_econ, x='VClass', y='comb', color=base_color, errwidth=0)
plt.xticks(rotation=15)
plt.ylabel('Avg. Combined Fuel Eff. (mpg)')
plt.show()

# show std dev
sb.barplot(data=fuel_econ, x='VClass', y='comb', color=base_color, ci='sd')
plt.xticks(rotation=15)
plt.ylabel('Avg. Combined Fuel Eff. (mpg)')
plt.show()

# point plot
sb.pointplot(data=fuel_econ, x='VClass', y='comb', ci='sd')
plt.xticks(rotation=15)
plt.ylabel('Avg. Combined Fuel Eff. (mpg)')
plt.show()

# remove the lines
sb.pointplot(data=fuel_econ, x='VClass', y='comb', ci='sd', linestyles='')
plt.xticks(rotation=15)
plt.ylabel('Avg. Combined Fuel Eff. (mpg)')
plt.show()