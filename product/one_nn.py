import numpy as np

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

"""
dataset = np.array([[1,2],[3,1],[1,1]])
label = ['1', '2', '1']
test_point = np.array([4,2])
predict = predict_1nn(dataset, label, test_point)
print("predicted label: ", predict)

"""
