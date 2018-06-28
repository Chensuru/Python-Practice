#1.	APPL.csv是蘋果從2017年6月19日到2018年6 月19 日的歷史股價。
# 其欄位依序為日期（日.月.年）、開盤價、最高價、最低價、收盤價、
# 調整後的收盤價(Adj Close)及成交量。請利用調整後的收盤價(Adj Close)
# 計算每天相對於前一天的漲跌幅，即 (今日Adj Close-昨日Adj Close)/昨日Adj Close，
# 並以日期為x軸，漲跌幅為y軸，畫出折線圖。
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


share_price = pd.read_csv('AAPL.csv')
change_rate = pd.Series(share_price['Adj Close']).pct_change()
share_price['Date'] = pd.to_datetime(share_price['Date'])
mark_list = []
mark_list_x = []
for i in range(len(change_rate)):
    if abs(change_rate[i]) > 0.03:
        mark_list.append(change_rate[i])
        mark_list_x.append(share_price['Date'][i])

plt.plot(share_price['Date'], change_rate)
plt.xlabel('Date')
plt.ylabel('Change Rate')

plt.scatter(mark_list_x[0:11], mark_list[0:11], color='r', marker='o', s=200)
plt.show()


