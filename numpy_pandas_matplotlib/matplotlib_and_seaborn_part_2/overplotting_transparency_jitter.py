import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

pd.set_option('display.max_columns', None)

fuel_econ = pd.read_csv('fuel-econ.csv')

sb.regplot(data=fuel_econ, x='year', y='comb', x_jitter=0.3, scatter_kws={'alpha':1/20})
ticks = range(2012, 2020)
labels = ['{}'.format(x) for x in ticks]
plt.xticks(ticks, labels)
plt.show()