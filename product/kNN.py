import numpy as np
from numpy import tile
import operator


def predict_kNN(dataset, labels, test_point, k):
  euclidean_distance = np.linalg.norm(dataset - test_point, axis = 1)
  print("distances: ", euclidean_distance)
  sort_indicies = np.argsort(euclidean_distance)
  k_indicies = sort_indicies[:k]
  print("k indicies: ", k_indicies)
  k_nearest_labels = labels[k_indicies]
  print("k nearest labels", k_nearest_labels)
  
 
data = np.array([[1,2],[2,2],[4,3],[3,1],[5,2], [1,1],[2,3]])
label = np.array(['A','B','A','C','B','A','C'])
test_point = [3,2]
prediction = predict_kNN(data, label, test_point, k=3)
print("Predicted label: ", prediction)
