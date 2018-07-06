import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

t = np.arange(0, 5, 0.01)
x = pd.period_range(pd.datetime.now(), periods=200, freq='d')
x = tuple(x.to_timestamp().to_pydatetime())
a1 = np.sin(2*np.pi*t)
a2 = np.sin(4*np.pi*t)
a3 = np.exp(-t)


y_label = ('foo', 'bra', 'xie')
y_color = ('b', 'g', 'r')
figure, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True, figsize=(8, 8))
label_data = zip(a3, a1, y_label, y_color)
print(type(a1))
print(a1)
for i, ld in enumerate(label_data):
    ax = axes[i]
    print(ld[0])
    ax.plot(ld[0], ld[1], label=ld[2], color=ld[3])
    ax.set_ylabel('Sum')
    ax.legend(loc='upper left', framealpha=0.5, prop={'size': 'small'})

plt.show()

