import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

fuel_econ = pd.read_csv('fuel-econ.csv')

g = sb.JointGrid(data=fuel_econ, x='displ', y='comb')
g.plot_joint(plt.scatter)
g.plot_marginals(sb.rugplot, height=0.25)

plt.show()
