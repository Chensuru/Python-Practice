import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#iris_data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
iris_data = pd.read_csv('iris.data.txt', header=None, encoding='utf-8', sep=',')
iris_data.columns = ['special length', 'special width', 'petal length', 'petal width', 'species']
iris_species = iris_data.groupby('species').size()
iris_species_group =iris_data.groupby('species')

#DataFrame的一開始資料探索
print(iris_data.head(5))
print(iris_data.info()) #Check if any null exits
print(iris_data.describe())
print(iris_data.ix[:24, 'species'])

#Series的取法
print(iris_data.groupby('species').size())
print(iris_species[2])
print(iris_species.tolist())
print(iris_data.groupby('species').groups)

for count in iris_species:
    print(count)

print(iris_species_group['special length'].agg([np.mean, np.sum, np.std]))
print(iris_species_group.agg({'special length': [np.mean, np.sum, np.std, min, max], 'special width': [np.mean, np.sum, np.std]}))

t = np.arange(0, 5, 0.01)
a1 = np.sin(2*np.pi*t)
a2 = np.sin(4*np.pi*t)
a3 = np.exp(-t)

ax1 = plt.subplot(311)
plt.setp(ax1.get_xticklabels(), fontsize=6)
plt.plot(t, a1)
plt.title('a1')

ax2 = plt.subplot(312, sharex=ax1, sharey=ax1)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.plot(t, a2)
plt.title('a2')

ax3 = plt.subplot(313)
plt.setp(ax3.get_xticklabels(), fontsize=6)
plt.plot(t, a3)
plt.xlim(0, 8)
plt.title('a3')
plt.show()


