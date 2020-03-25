import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

fuel_econ = pd.read_csv('fuel-econ.csv')

# Task: What is the relationship between the size of a car and the size of its engine? The cars in this dataset are
# categorized into one of five different vehicle classes based on size. Starting from the smallest, they are:
# {Minicompact Cars, Subcompact Cars, Compact Cars, Midsize Cars, and Large Cars}. The vehicle classes can be found in
# the 'VClass' variable, while the engine sizes are in the 'displ' column (in liters). Hint: Make sure that the order
# of vehicle classes makes sense in your plot!

sedan_classes = ['Minicompact Cars', 'Subcompact Cars', 'Compact Cars', 'Midsize Cars', 'Large Cars']
vclasses = pd.api.types.CategoricalDtype(ordered=True, categories=sedan_classes)
fuel_econ['VClass'] = fuel_econ['VClass'].astype(vclasses)

sb.violinplot(data=fuel_econ, x='VClass', y='displ', color=sb.color_palette()[0])
plt.xticks(rotation=15)
plt.show()
