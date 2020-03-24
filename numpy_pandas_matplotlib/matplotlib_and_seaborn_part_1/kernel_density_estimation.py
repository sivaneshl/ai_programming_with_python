import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np

data = [0.0, 3.0, 4.5, 8.0]
plt.figure(figsize=[12, 5])

# left plot: showing kde lumps with the default settings
plt.subplot(1, 3, 1)
sb.distplot(data, hist=False, rug=True, rug_kws={'color': 'r'})

# central plot: kde with narrow bandwidth to show individual probability lumps
plt.subplot(1, 3, 2)
sb.distplot(data, hist=False, rug=True, rug_kws={'color': 'r'}, kde_kws={'bw': 1})

# right plot: choosing a different, triangular kernel function (lump shape)
plt.subplot(1, 3, 3)
sb.distplot(data, hist=False, rug=True, rug_kws={'color': 'r'}, kde_kws={'bw': 1, 'kernel': 'gau'})

plt.show()