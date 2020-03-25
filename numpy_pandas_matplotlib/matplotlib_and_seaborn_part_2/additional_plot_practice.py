import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

fuel_econ = pd.read_csv('fuel-econ.csv')

# Task 1: Plot the distribution of combined fuel mileage (column 'comb', in miles per gallon) by manufacturer (column
# 'make'), for all manufacturers with at least eighty cars in the dataset. Consider which manufacturer order will convey
# the most information when constructing your final plot. Hint: Completing this exercise will take multiple steps! Add
# additional code cells as needed in order to achieve the goal.

makes_select = fuel_econ['make'].value_counts().reset_index(name='count').query('count>80')['index'].to_list()
fuel_econ = fuel_econ[fuel_econ['make'].isin(makes_select)]
makes_ordered = fuel_econ[['make','comb']].groupby('make')['comb'].mean().sort_values(ascending=False).reset_index()['make'].to_list()

bins = np.arange(fuel_econ['comb'].min(), fuel_econ['comb'].max()+2, 2)
g = sb.FacetGrid(data=fuel_econ, col='make', col_wrap=6, col_order=makes_ordered)
g.map(plt.hist, 'comb', bins=bins)
plt.show()

# Task 2: Continuing on from the previous task, plot the mean fuel efficiency for each manufacturer with at least 80
# cars in the dataset.
sb.barplot(data=fuel_econ, y='make', x='comb', color=sb.color_palette()[0], order=makes_ordered, ci='sd')
plt.xlabel('Avg. Combined Fuel Eff. (mpg)')
plt.ylabel('Make')
plt.show()