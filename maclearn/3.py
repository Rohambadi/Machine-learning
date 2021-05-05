from sklearn import datasets
import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


iris = datasets.load_iris()
print(iris.data.shape)
print(iris.feature_names)
print(iris.target_names)


iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
print(iris_df)


knn = KNeighborsClassifier(n_neighbors=6, metric='minkowski', p=2)
x = iris.data
y = iris.target
knn.fit(x, y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
print(x_train.shape)
print(x_test.shape)


knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(x_train, y_train)
y_predict = knn.predict(x_test)
print(y_predict)
print(knn.score(x_test, y_test))

dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
predict_dtc = dtc.predict(x_test)
metrics.accuracy_score(y_test, predict_dtc)
print(dtc.score(x_test, y_test))
