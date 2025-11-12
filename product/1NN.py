import numpy as np

#import and load iris dataset 
from sklearn.datasets import load_iris
iris = load_iris()
X = iris['data']
y = iris['target']

#split the dataset, extracts 75% for training and 25% for testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 0)

print("Original dataset shape X:",X.shape)
print("Training data shape X_train:",X_train.shape)
print("Testing data shape X_test:",X_test.shape)

#1NN Function
def predict_1nn (dataset, label, test_point):
  """
  Compute the Euclidean distance between the test_point and each row in the dataset
  For example, datasets are [1,2] and [3,1] and test_point is [4,2].
  
  1st: (dataset- testpoint),
  Row 0: [1,2] - [4,2] = [-3,0]
  Row 1: [3,1] - [4,2] = [-1,-1]
  
  2nd: np.linalg.norm numpy function compute the norm length(squared each then sum all then sqrt that sum)
  """
  euclidean_distance = np.linalg.norm(dataset - test_point, axis = 1)
  target = np.argmin(euclidean_distance) #return the index of the smallest value
  return label[target]

dataset = np.array([[1,2],[3,1],[1,1]])
label = ['1', '2', '1']
test_point = np.array([4,2])
predict = predict_1nn(dataset, label, test_point)
print("predicted label: ", predict)

#empty list to store the predictions
evaluation_arr = []
#loop through every item in test set as
#the 1NN function only predict one test point at a time  
for test_point in X_test:
  predict = predict_1nn(X_train, y_train, test_point)
  evaluation_arr.append(predict)
  
y_pred = np.array([evaluation_arr])
#y_pred == y_test get boolean array [True,False..] where True is 1 and False is 0
#mean takes the average.(1+1+0+1)/4
score = np.mean(y_pred==y_test) 
print("Predicted labels: ", y_pred)
print("Actual labels: ", y_test)
print("Accuray score: ", score)

print("target names: ", iris['target_names'])
print("iris data: ", iris['data'][:5])

#predict label with new data
X_new = np.array([6.0, 4.1, 5.0, 1.2])
prediction = predict_1nn(X_train, y_train, X_new)
print("Predicted label of the new data: ", prediction)
print("Target name of the new data: ", iris['target_names'][prediction])


