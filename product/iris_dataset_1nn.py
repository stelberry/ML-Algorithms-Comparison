import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#import the 1NN function from one_nn.py
from one_NN import predict_1nn

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


#create figure with 2 side-by-side plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

#plot actual labels
for i in range(3): #loop 3 times since there are 3 types of species (0 = Setosa, 1 = Versicolor, and 2 = Virginica).
    #create a filter to find matching the current species (eg.[True, True, False,...])
    is_current_species = y_test == i 
    #get sepal length(col 0), only for the current one(is_current_species)
    #get sepal width(col 1), only for the current one(is_current_species)
    ax1.scatter(X_test[is_current_species, 0], X_test[is_current_species, 1], 
                label=iris['target_names'][i], alpha=0.7)
                
ax1.set_title("Actual Labels")
ax1.set_xlabel(iris['feature_names'][0])
ax1.set_ylabel(iris['feature_names'][1])
ax1.legend()

#plot predicted labels
#use the same logic
for i in range(3):
    is_current_species = y_pred == i
    ax2.scatter(X_test[is_current_species, 0], X_test[is_current_species, 1], 
                label=iris['target_names'][i], alpha=0.7)
                
ax2.set_title("Predicted Labels")
ax2.set_xlabel(iris['feature_names'][0])
ax2.set_ylabel(iris['feature_names'][1])
ax2.legend()

plt.show()
