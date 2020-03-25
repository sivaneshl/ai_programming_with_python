import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

fuel_econ = pd.read_csv('fuel-econ.csv')

sedan_classes = ['Minicompact Cars', 'Subcompact Cars', 'Compact Cars', 'Midsize Cars', 'Large Cars']
vclasses = pd.api.types.CategoricalDtype(ordered=True, categories=sedan_classes)
fuel_econ['VClass'] = fuel_econ['VClass'].astype(vclasses)


fuel_econ['trans_type'] = fuel_econ['trans'].apply(lambda x: x.split()[0])

ct_counts = fuel_econ.groupby(['VClass', 'trans_type']).size()
ct_counts = ct_counts.reset_index(name='count')
ct_counts = ct_counts.pivot(index='VClass', columns='trans_type', values='count')
# print(ct_counts)

sb.heatmap(ct_counts, annot=True, fmt='d')
plt.show()

sb.countplot(data=fuel_econ, x='VClass', hue='trans_type')
plt.xticks(rotation=15)
plt.show()
