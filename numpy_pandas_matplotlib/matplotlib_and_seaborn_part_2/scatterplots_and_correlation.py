import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

pd.set_option('display.max_columns', None)

fuel_econ = pd.read_csv('fuel-econ.csv')
print(fuel_econ.shape)
# print(fuel_econ.head())

# using matplotlib
plt.scatter(data=fuel_econ, x='displ', y='comb')
plt.ylabel('Displacement (l)')
plt.xlabel('Combined Fuel Eff. (mpg)')
plt.show()


# using seaborn
sb.regplot(data=fuel_econ, x='displ', y='comb')
plt.ylabel('Displacement (l)')
plt.xlabel('Combined Fuel Eff. (mpg)')
plt.show()


# log trans
def log_trans(x, inverse=False):
    if not inverse:
        return np.log10(x)
    else:
        return np.power(10, x)


sb.regplot(fuel_econ['displ'],
           fuel_econ['comb'].apply(log_trans))
ticks = [10, 20, 30, 40, 50, 60]
plt.yticks(log_trans(ticks), ticks)
plt.show()

