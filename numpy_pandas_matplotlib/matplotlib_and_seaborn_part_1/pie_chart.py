import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

pokemon = pd.read_csv('pokemon.csv')

sorted_counts = pokemon['generation_id'].value_counts()
plt.pie(sorted_counts, labels=sorted_counts.index, startangle=90, counterclock=False)
plt.axis('square')
plt.show()


# donut plot
plt.pie(sorted_counts, labels=sorted_counts.index, startangle=90, counterclock=False,
        wedgeprops={'width': 0.4})
plt.axis('square')
plt.show()
