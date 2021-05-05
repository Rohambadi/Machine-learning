import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sb

smartphones = pd.read_csv('smartphones.csv')
count = smartphones.Ram.value_counts()
category = count.index


def ECDF(data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1)/n
    return x, y


x1, y1 = ECDF(smartphones.inch)

plt.figure(figsize=(11, 8))
plt.scatter(x1, y1, s=80)
plt.margins(0.05)
plt.xlabel("inch", fontsize=15)
plt.ylabel("ECDF", fontsize=15)
plt.show()
