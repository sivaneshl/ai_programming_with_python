import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

fuel_econ = pd.read_csv('fuel-econ.csv')

# pre-processing: count and sort by the number of instances of each category
sorted_counts = fuel_econ['VClass'].value_counts()

# establish the Figure
plt.figure(figsize = [12, 5])

# left plot: pie chart
plt.subplot(1, 2, 1)
plt.pie(sorted_counts, labels=sorted_counts.index, startangle=90, counterclock=False)
plt.axis('square')

# right plot: horizontally stacked bar
plt.subplot(1, 2, 2)
baseline = 0
for i in range(sorted_counts.shape[0]):
    plt.barh(y=1, width=sorted_counts[i], left=baseline)
    baseline+=sorted_counts[i]
plt.legend(sorted_counts.index)
plt.ylim([0, 2])
plt.show()