import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd

def calu(age, divide_group):
    return age//divide_group

adult_data = pd.read_table('adult.data.txt', header=None, encoding='utf-8', sep=',')
adult_data_age = adult_data[0].tolist()

age_group_list = []
for i in range(len(adult_data_age)):
    age_group_list.append(calu(adult_data_age[i], 10)*10)

print(adult_data_age)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
ax.hist(age_group_list, bins=8, color='green', alpha=0.5)
ax.set_xlabel('haha')
ax.set_ylabel('age')
#ax.set_xticks([0, 100])
#ax.set_xticklabels(['1', '2', '3', '4'])
plt.show()


