import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

#讀取資料
iris_data = pd.read_csv('iris.data.txt', header=None)

#拿到資料查看
iris_data.describe()
iris_data.shape
#print(iris_data.head(10))
#iris_data.shape()
#print(iris_data.groupby(4).count())

#轉換預測資料(將文字轉成0/1)
iris_feature_column = iris_data.ix[:, 0:3]
iris_data_predict = iris_data.ix[:, 4]
x = np.array(iris_feature_column)
y = np.array(iris_data_predict)
y_lenum = LabelEncoder().fit_transform(y)

#KNN算法，random_state=0，混淆矩陣
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
knn_score = classifier.score(x_test, y_test)
acc_score = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
#print(classifier.fit(x_train, y_train))
#print(cm, knn_score, acc_score)

#優化模型調參
k_list = list(range(1, 50, 1))
cv_scores = []
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

#print(cv_scores)
best_k = k_list[cv_scores.index(max(cv_scores))]
#print(best_k)

#畫成圖形
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.set_xlabel('k')
ax.set_ylabel('cv_scores')
ax.plot(k_list, cv_scores, color='b')
ax.scatter(best_k, max(cv_scores), color='r', marker='o', s=200)
plt.show()
