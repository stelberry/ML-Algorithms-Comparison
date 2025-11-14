import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#import the 1NN function from one_nn.py
from kNN import predict_knn

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
  predict = predict_knn(X_train, y_train, test_point, k=3 )
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
prediction = predict_knn(X_train, y_train, X_new, k= 5)
print("Predicted label of the new data: ", prediction)
print("Target name of the new data: ", iris['target_names'][prediction])

# array to store each species accuracy
species_accuracies = []
names = iris['target_names']
count = 0

# Bar Plot of Accuracy per Species(Flower Type)
# loop 3 times (0 = Setosa, 1 = Versicolor, 2 = Virginica)
for i in range(3):
  # filter the current_species, return boolean array (eg.[True, True, False,...])
  is_current_species = y_test == i
  # same logic as calculating accuracy score from above
  accuracy = np.mean(y_test[is_current_species] == y_pred[is_current_species])
  # store 3 scores of each species
  species_accuracies.append(accuracy)

plt.figure(figsize=(8, 5))
bars = plt.bar(names, species_accuracies, color=['violet', 'purple', 'plum'])

plt.title("Prediction Accuracy by Flower Type")
plt.ylabel("Accuracy (0-1)")
plt.ylim(0, 1.1) # Make space for bar lables
plt.bar_label(bars, fmt='%.2f') # Show numbers on top

plt.show()
  