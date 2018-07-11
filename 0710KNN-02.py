import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

#乳癌診斷
def data_trans(data_list, column_list):
    return pd.DataFrame(data_list, columns=column_list)

def data_result():
    cancer_yes = np.sum(cancer['target'] > 0)
    cancer_no = np.sum(cancer['target'] < 1)
    print(pd.Series(np.array(cancer_no, cancer_yes), index=['malignant','benign']))

cancer = load_breast_cancer()
#print(cancer.DESCR)
#print(cancer['feature_names']) # 打印数据集的描述
#print(cancer.keys())
#print(cancer['data'])
#print(cancer['target_names'])

cancer_columns = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
 'mean smoothness', 'mean compactness', 'mean concavity',
 'mean concave points', 'mean symmetry', 'mean fractal dimension',
 'radius error', 'texture error', 'perimeter error', 'area error',
 'smoothness error', 'compactness error', 'concavity error',
 'concave points error', 'symmetry error', 'fractal dimension error',
 'worst radius', 'worst texture', 'worst perimeter', 'worst area',
 'worst smoothness', 'worst compactness', 'worst concavity',
 'worst concave points', 'worst symmetry', 'worst fractal dimension']
cancer_dataframe = data_trans(cancer['data'], cancer_columns)
cancer_result = data_trans(cancer['target'], ['cancer_result'])
print(cancer_dataframe.head(10))
#data_result()
#print(pd.DataFrame(cancer['data'], columns=cancer_columns))
#print(np.sum(cancer['target']))
#print(np.sum(cancer['target'] > 0))
#print(cancer_result)

cancer_size = cancer_result.groupby('cancer_result').size()
#print(cancer_size[0])
#print(cancer_size[1])
X = cancer_dataframe[['mean radius', 'mean texture', 'mean perimeter', 'mean area',
'mean smoothness', 'mean compactness', 'mean concavity',
'mean concave points', 'mean symmetry', 'mean fractal dimension',
'radius error', 'texture error', 'perimeter error', 'area error',
'smoothness error', 'compactness error', 'concavity error',
'concave points error', 'symmetry error', 'fractal dimension error',
'worst radius', 'worst texture', 'worst perimeter', 'worst area',
'worst smoothness', 'worst compactness', 'worst concavity',
'worst concave points', 'worst symmetry', 'worst fractal dimension'] ]
x_train, x_test, y_train, y_test = train_test_split(X, cancer['target'], test_size=0.2, random_state=0)
print("shape", x_train.shape, x_test.shape, y_train.shape, y_test.shape)
classifier = KNeighborsClassifier(n_neighbors=10)
knn_fit = classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
knn_score = classifier.score(x_test, y_test)
acc_score = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("fit", knn_fit)
print("cn", cm, knn_score, acc_score)


#優化模型調參
k_list = list(range(1, 50, 1))
cv_scores = []
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())


best_k = k_list[cv_scores.index(max(cv_scores))]
print("best_k:", best_k)
print("cv_scores", cv_scores[best_k])


means = cancer_dataframe.mean().values.reshape(1, -1)
x_text_result_df = pd.DataFrame(x_test)
x_text_result_df['predition'] = y_pred
x_text_result_df['real'] = y_test

print("means", means)
print("prediction", x_text_result_df)

#畫成圖形
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.set_xlabel('k')
ax.set_ylabel('cv_scores')
ax.plot(k_list, cv_scores, color='b')
ax.scatter(best_k, max(cv_scores), color='r', marker='o', s=200)
plt.show()