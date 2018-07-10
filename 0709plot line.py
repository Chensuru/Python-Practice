import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = pd.period_range(pd.datetime.now(), periods=200, freq='1h20min')
x = x.to_timestamp().to_pydatetime()
y = np.random.randn(200, 3).cumsum(0)
k = np.random.seed(1234)
#x = x.to_timestamp().to_pydatetime()
#print(pd.datetime.now())
#print(type(x))
#print(type(x.to_timestamp().to_pydatetime()))
#print(x.to_timestamp().to_pydatetime())
#print(y)
#print(k)
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
ax_list = axs.plot(x, y)
print(type(ax_list))
axs.set_title('Show')
axs.set_xlabel('Date Time')
axs.set_ylabel('Haha')
axs.legend(ax_list, {'foo', 'b', 'cc'}, loc='best', framealpha=0.5, prop={'size':'small', 'family':'monospace'})
plt.figtext(0.095, 0.01, 'hahahah', ha='right', va='bottom')
#plt.xlabel('mm')

ax_small = fig.add_axes([0.15, 0.2, 0.3, 0.3])
#ax_small.set_xticks([])
ax_small.set_yticks([])
plt.tight_layout()
#plt.set_tight_layout()
#plt.gcf().set_tight_layout(True)
plt.savefig('fig.png',bbox_inches='tight')
plt.show()

