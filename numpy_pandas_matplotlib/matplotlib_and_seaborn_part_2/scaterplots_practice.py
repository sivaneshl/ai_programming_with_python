import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

fuel_econ = pd.read_csv('fuel-econ.csv')

# Task 1: Let's look at the relationship between fuel mileage ratings for city vs. highway driving, as stored in the
# 'city' and 'highway' variables (in miles per gallon, or mpg). Use a scatter plot to depict the data. What is the
# general relationship between these variables? Are there any points that appear unusual against these trends?
sb.regplot(data=fuel_econ, x='city', y='highway', scatter_kws={'alpha': 1/8})
plt.xlabel('City Mileage')
plt.ylabel('Highway Mileage')
plt.show()

# Task 2: Let's look at the relationship between two other numeric variables. How does the engine size relate to a
# car's CO2 footprint? The 'displ' variable has the former (in liters), while the 'co2' variable has the latter
# (in grams per mile). Use a heat map to depict the data. How strong is this trend?
bins_x = np.arange(0.6, fuel_econ['displ'].max()+0.5, 0.5)
bins_y = np.arange(29, fuel_econ['co2'].max()+50, 50)
plt.hist2d(data=fuel_econ, x='displ', y='co2', cmin=0.5, bins=[bins_x, bins_y], cmap='viridis_r')
plt.colorbar()
plt.xlabel('Displacement (l)')
plt.ylabel('CO2 (g/mi)')
plt.show()