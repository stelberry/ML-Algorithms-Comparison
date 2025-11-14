import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#import the 1NN function from one_nn.py
from one_nn import predict_1nn

"""
The dataset consists of 3 types of Iris flowers. 
We measure 4 for every 150 flower: Sepal Length, Sepal Width, Petal Length, and Petal Width.
X (data): measurements (e.g., 5.1, 3.5, 1.4, 0.2)
y (target): species name in int
0 = Setosa
1 = Versicolor
2 = Virginica
"""
iris = load_iris()
X = iris['data']
y = iris['target']

print("iris data: ", X[:5])
print("target names:", iris['target_names'])
print("feature names:",iris['feature_names'])

#split the dataset, extracts 75% for training and 25% for testing
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 0)

print("Original dataset shape X:",X.shape)
print("#150 flowers has 4 measurements")

print("Training data shape X_train:",X_train.shape)
print("Testing data shape X_test:",X_test.shape)

#empty list to store the predictions
evaluation_arr = []

#loop through every item in test set as
#the 1NN function only predict one test point at a time  
for test_point in X_test:
  predict = predict_1nn(X_train, y_train, test_point)
  evaluation_arr.append(predict)
  
y_pred = np.array(evaluation_arr)
#y_pred == y_test get boolean array [True,False..] where True is 1 and False is 0
#mean takes the average.(1+1+0+1)/4
score = np.mean(y_pred==y_test) 
error_rate = 1 - score

print("Predicted labels: ", y_pred)
print("Actual labels: ", y_test)
print("Accuray score: ", score)
print("Error rate:", error_rate)

#predict label with new data
X_new = np.array([6.0, 4.1, 5.0, 1.2])
prediction = predict_1nn(X_train, y_train, X_new)
print("Predicted label of the new data: ", prediction)
print("Target name of the new data: ", iris['target_names'][prediction])


