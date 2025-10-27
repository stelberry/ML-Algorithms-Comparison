import numpy as np

from sklearn.datasets import load_iris
iris = load_iris()
X = iris['data']
y = iris['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 0)

print("Original dataset shape: ",X.shape)
print("Training data shape: ",X_train.shape)
print("Testing data shape: ",X_test.shape)

def predict_1nn (dataset, label, test_point):
  euclidean_distance = np.linalg.norm(dataset - test_point, axis = 1)
  print("euclidean_distance", euclidean_distance)
  target = np.argmin(euclidean_distance)
  return label[target]

"""dataset = np.array([[1,2],[3,1],[1,1]])
label = ['1', '2', '1']
test_point = np.array([4,2])
predict = predict_1nn(dataset, label, test_point)
print("predicted label: ", predict)"""

evaluation_arr = []
for test_point in X_test:
  predict = predict_1nn(X_train, y_train, test_point)
  evaluation_arr.append(predict)
  
y_pred = np.array([evaluation_arr])
score = np.mean(y_pred==y_test)
print("Predicted labels: ", y_pred)
print("Actual labels: ", y_test)
print("Accuray score: ", score)

