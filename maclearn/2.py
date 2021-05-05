from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


iris = datasets.load_iris()
print(iris.data.shape)
print(iris.feature_names)
print(iris.target_names)
# print(iris.DESCR)


iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(iris_df)
iris_df['target'] = iris.target
print(iris_df)

# visual EDA
# pd.plotting.scatter_matrix(iris_df,c=iris.target,figsize=[11,11],s=150)
# plt.show()
'''x=iris.data[:,[2,3]]
y=iris.target'''
# plt.scatter(x[:,0],x[:,1],c=y)
# plt.show()


knn = KNeighborsClassifier(n_neighbors=6, metric='minkowski', p=2)
x = iris.data
y = iris.target
knn.fit(x, y)
'''x_new=np.array([[5,3,1,0.2]])
y_new=knn.predict(x_new)
print(y_new)'''


# train and test:

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
print(x_train.shape)
print(x_test.shape)


knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(x_train, y_train)
y_predict = knn.predict(x_test)
print(y_predict)
print(knn.score(x_test, y_test))


neighbors = np.arange(1, 30)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))


for i, k in enumerate(neighbors):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(x_train, y_train)
    train_accuracy[i] = knn_model.score(x_train, y_train)
    test_accuracy[i] = knn_model.score(x_test, y_test)

plt.plot(neighbors, train_accuracy, label='train_accuracy')
plt.plot(neighbors, test_accuracy, label='test_accuracy')
plt.legend()
plt.xlabel("number of neighbors")
plt.ylabel("Accuracy")
plt.show()


dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
predict_dtc = dtc.predict(x_test)


metrics.accuracy_score(y_test, predict_dtc)
print(dtc.score(x_test, y_test))
