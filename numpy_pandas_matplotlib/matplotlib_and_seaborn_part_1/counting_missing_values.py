import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

pokemon = pd.read_csv('pokemon.csv')

na_counts = pokemon.isna().sum()
print(na_counts)

base_color = sb.color_palette()[0]
sb.barplot(na_counts.index.values, na_counts, color=base_color)
plt.xticks(rotation=90)
plt.show()