import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

fuel_econ = pd.read_csv('fuel-econ.csv')

# Task: Use a plot to explore whether or not there differences in recommended fuel type depending on the vehicle class.
# Only investigate the difference between the two main fuel types found in the 'fuelType' variable: Regular Gasoline
# and Premium Gasoline. (The other fuel types represented in the dataset are of much lower frequency compared to the
# main two, that they'll be more distracting than informative.) Note: The dataset as provided does not retain any of
# the sorting of the 'VClass' variable, so you will also need to copy over any code you used previously to sort the
# category levels.

fuel_econ = fuel_econ[fuel_econ['fuelType'].isin(['Regular Gasoline','Premium Gasoline'])]

sedan_classes = ['Minicompact Cars', 'Subcompact Cars', 'Compact Cars', 'Midsize Cars', 'Large Cars']
vclasses = pd.api.types.CategoricalDtype(ordered=True, categories=sedan_classes)
fuel_econ['VClass'] = fuel_econ['VClass'].astype(vclasses)

sb.countplot(data=fuel_econ, x='VClass', hue='fuelType')
plt.xticks(rotation=15)
ax = plt.gca()
ax.legend(loc=4, ncol=1, title='Fuel Type', framealpha=1)
plt.show()
