import numpy as np
import matplotlib.pyplot as plt

# Define vector v
v = np.array([1, 1])

# Define vector w
w = np.array([-2, 2])

# Plots vector v(blue arrow) and vector w(cyan arrow) with red dot at origin (0,0) using Matplotlib

# Creates axes of plot referenced 'ax'
ax = plt.axes()
# Plots red dot at origin (0,0)
ax.plot(0, 0, 'or')
# Plots vector v as blue arrow starting at origin 0,0
plt.arrow(0, 0, *v, color='b', linewidth=2.5, head_width=0.25, head_length=0.35)
# Plots vector w as cyan arrow starting at origin 0,0
plt.arrow(0, 0, *w, color='c', linewidth=2.5, head_width=0.25, head_length=0.35)
# Sets limit for plot for x-axis
plt.xlim(-3, 2)
# Set major ticks for x-axis
major_xticks = np.arange(-3, 2)
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

# Creates axes of plot referenced 'ax'
ax = plt.axes()
# Plots red dot at origin (0,0)
ax.plot(0, 0, 'or')
# Plots vector v as blue arrow starting at origin 0,0
plt.arrow(0, 0, *v, color='b', linewidth=2.5, head_width=0.25, head_length=0.35)
# Plots vector w as cyan arrow starting at origin 0,0
plt.arrow(v[0], v[1], *w, color='c', linewidth=2.5, head_width=0.25, head_length=0.35, linestyle='dotted')
# Sets limit for plot for x-axis
plt.xlim(-3, 2)
# Set major ticks for x-axis
major_xticks = np.arange(-3, 2)
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

# Adding Two Vectors and Plotting Results

# DONE 1:. Define vector vw by adding vectors v and w
vw = v+w
# Creates axes of plot referenced 'ax'
ax = plt.axes()
# Plots red dot at origin (0,0)
ax.plot(0, 0, 'or')
# Plots vector v as blue arrow starting at origin 0,0
plt.arrow(0, 0, *v, color='b', linewidth=2.5, head_width=0.25, head_length=0.35)
# Plots vector w as cyan arrow starting at origin 0,0
plt.arrow(v[0], v[1], *w, color='c', linewidth=2.5, head_width=0.25, head_length=0.35, linestyle='dotted')
# DONE 2:. Plot vector vw as black arrow (color='k') with 3.5 linewidth (linewidth=3.5) starting vector v's origin (0,0)
plt.arrow(0, 0, *vw, color='k', linewidth=3.5, head_width=0.25, head_length=0.35)
# Sets limit for plot for x-axis
plt.xlim(-3, 2)
# Set major ticks for x-axis
major_xticks = np.arange(-3, 2)
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