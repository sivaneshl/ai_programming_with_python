import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

fuel_econ = pd.read_csv('fuel-econ.csv')

year_order = sorted(list(fuel_econ['year'].unique()))
vclass_order = ['Minicompact Cars', 'Subcompact Cars', 'Compact Cars', 'Midsize Cars', 'Large Cars']

artists = []
baselines = np.zeros(len(year_order))
year_counts = fuel_econ['year'].value_counts()

# for each second-variable category:
for i in range(len(vclass_order)):
    # isolate the counts of the first category,
    vclass = vclass_order[i]
    inner_counts = fuel_econ[fuel_econ['VClass']==vclass]['year'].value_counts()
    inner_props = inner_counts / year_counts
    # then plot those counts on top of the accumulated baseline
    bars = plt.bar(x=np.arange(len(year_order)),
                   height=inner_props[year_order],
                   bottom=baselines)
    artists.append(bars)
    baselines += inner_props[year_order]
plt.xticks(np.arange(len(year_order)), year_order)
plt.legend(reversed(artists), reversed(vclass_order), bbox_to_anchor=(1.08, 0.5), loc=8, framealpha=1)
plt.show()