

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

'''
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

print(iris_X)
print(iris_X[:2,:])
print(iris_y)

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)

print(y_train)
print('*'*100)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

print(knn.predict(X_test))
print(y_test)

'''

loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target

model = LinearRegression()
model.fit(data_X, data_y)

print(model.predict(data_X[:4,:]))
print(data_y[:4])


X, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=10)

plt.scatter(X, y)

plt.show()