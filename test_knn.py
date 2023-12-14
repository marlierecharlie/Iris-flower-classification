import pytest
from iris_selfmade_KNN import KNN
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

def test_knn_predict():
  data = KNN()
  data.fit(x_train, y_train)
  answer = data.predict(x_test)
  assert isinstance(answer, list)


a=test_knn_predict()
print(a)
