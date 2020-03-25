import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

fuel_econ = pd.read_csv('fuel-econ.csv')

df_window = fuel_econ.sort_values('city').rolling(15)
x_winmean = df_window.mean()['city']
y_median = df_window.median()['highway']
y_q1 = df_window.quantile(.25)['highway']
y_q3 = df_window.quantile(.75)['highway']

base_color = sb.color_palette()[0]
line_color = sb.color_palette('dark')[0]
plt.scatter(data=fuel_econ, x='city', y='highway')
plt.errorbar(x=x_winmean, y=y_median, c=line_color)
plt.errorbar(x=x_winmean, y=y_q1, c=line_color, linestyle='--')
plt.errorbar(x=x_winmean, y=y_q3, c=line_color, linestyle='--')

plt.xlabel('city')
plt.ylabel('highway')
plt.show()