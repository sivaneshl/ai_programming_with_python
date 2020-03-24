import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

pokemon = pd.read_csv('pokemon.csv')

fig = plt.figure()
ax = fig.add_axes([.125, .125, .775, .775])
ax.hist(data=pokemon, x='speed', bins=20)
plt.show()

fig = plt.figure()
ax = fig.add_axes([.125, .125, .775, .775])
base_color = sb.color_palette()[0]
sb.countplot(data=pokemon, x='generation_id', color=base_color, ax=ax)
plt.show()

# subplots
plt.figure(figsize=[10, 5])
# 1st plot
plt.subplot(1, 2, 1)    # 1 row, 2 cols, subplot 1
bins = np.arange(0, pokemon['speed'].max()+5, 5)
plt.hist(data=pokemon, x='speed', bins=bins)
# 2nd plot
plt.subplot(1, 2, 2)    # 1 row, 2 cols, subplot 2
bins = np.arange(0, pokemon['speed'].max()+1, 1)
plt.hist(data=pokemon, x='speed', bins=bins)
plt.show()


# subplots and axes
fig, axes = plt.subplots(3, 4)  # grid of 3X4 subplots
axes = axes.flatten()   # reshape from 3x4 array into 12-element vector
for i in range(12):
    plt.sca(axes[i])    # set the current Axes
    plt.text(0.5, 0.5, i+1)     # print conventional subplot index number to middle of Axes
plt.show()