import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

fuel_econ = pd.read_csv('fuel-econ.csv')

year_order = sorted(list(fuel_econ['year'].unique()))
vclass_order = ['Minicompact Cars', 'Subcompact Cars', 'Compact Cars', 'Midsize Cars', 'Large Cars']

plt.figure(figsize=[12, 5])

# left plot: clustered bar chart, absolute counts
plt.subplot(1, 2, 1)
sb.countplot(data=fuel_econ, x='year', hue='VClass', order=year_order, hue_order=vclass_order)
plt.legend(loc=8, framealpha=1)

# right plot: stacked bar chart, absolute counts
plt.subplot(1, 2, 2)
baselines = np.zeros(len(year_order))
# for each second-variable category:
for i in range(len(vclass_order)):
    # isolate the counts of the first category,
    vclass = vclass_order[i]
    inner_counts = fuel_econ[fuel_econ['VClass']==vclass]['year'].value_counts()
    # then plot those counts on top of the accumulated baseline
    plt.bar(x=np.arange(len(year_order)), height=inner_counts[year_order], bottom=baselines)
    baselines += inner_counts[year_order]
plt.xticks(np.arange(len(year_order)), year_order)
plt.legend(vclass_order, loc=8, framealpha=1)
plt.show()