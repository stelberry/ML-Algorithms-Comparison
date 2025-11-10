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
#calculate the euclidean distance 
  euclidean_distance = np.linalg.norm(dataset - test_point, axis = 1) #Compute norm across each row
  target = np.argmin(euclidean_distance) #return the index of the smallest value
  return label[target]

"""dataset = np.array([[1,2],[3,1],[1,1]])
label = ['1', '2', '1']
test_point = np.array([4,2])
predict = predict_1nn(dataset, label, test_point)
print("predicted label: ", predict)"""

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


