import numpy as np
import matplotlib.pyplot as plt

# Define vector v
v = np.array([1, 1])

# Define scalar a
a = 3

# TODO 1.: Define vector av - as vector v multiplied by scalar a
av = a*v

# Plots vector v as blue arrow with red dot at origin (0,0) using Matplotlib
ax = plt.axes()
# Plots red dot at origin (0,0)
ax.plot(0, 0, 'or')
# Plots vector v as blue arrow starting at origin 0,0
ax.arrow(0, 0, *v, color='b', linewidth=2.5, head_width=0.25, head_length=0.35)

# TODO 2.: Plot vector av as dotted (linestyle='dotted') vector of cyan color (color='c')
# using ax.arrow() statement above as template for the plot
ax.arrow(0, 0, *av, linewidth=2.5, head_width=0.25, head_length=0.35, linestyle='dotted', color='c')

# Sets limit for plot for x-axis
plt.xlim(-2, 4)
# Set major ticks for x-axis
major_xticks = np.arange(-2, 4)
ax.set_xticks(major_xticks)
# Sets limit for plot for y-axis
plt.ylim(-1, 4)
# Set major ticks for y-axis
major_yticks = np.arange(-1, 4)
ax.set_yticks(major_yticks)

# Creates gridlines for only major tick marks
plt.grid(b=True, which='major')

# Displays final plot
plt.show()