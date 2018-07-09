import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#多張子圖練習
x = np.linspace(-1, 1, 50)
y1 = 2*x + 1
y2 = 2**x + 1
y3 = 3**x + 1

#fig1 = plt.figure(num=1, figsize=(8, 4))
#fig2 = plt.figure(num=2, figsize=(8, 4))

fig, axs = plt.subplots(ncols=1, nrows=2, sharex=True)
y = list()
y.append(y1)
y.append(y2)
data_set_color = ('b', 'g')
data_set_yname = ('y1 test', 'y2 test')
data_set = zip(y, data_set_color, data_set_yname)
#print(x)
#print(type(data_set))
#print(len(axs))
#[(0, ('b', 'y1 test')), (1, ('g', 'y2 test'))]
print(enumerate(data_set))
for i, ld in enumerate(data_set):
    #print(i, ld, ld[1])
    ax = axs[i]
    ax.set_xlim([0, 1])
    #ax.set_xticks([0, 250, 500, 750, 1000])
    ax.set_xticklabels(['one', 'two', 'three', 'four', 'five', 'six'], rotation=30, fontsize='small')
    stri = 'Mt'+str(i)+ld[1]
    ax.set_title(stri)
    ax.set_xlabel('Stages')
    ax.set_ylabel('Sun')
    ax.plot(x, ld[0], label=ld[2], color=ld[1])
    ax.legend(loc='upper left', framealpha=0.5, prop={'size': 'small'})


#plt.plot(x, y1)
axs[0].plot(x, y3, color='red', linewidth=2.0, linestyle='--')

plt.show()

