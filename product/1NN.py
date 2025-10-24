import numpy as np
def predict_1nn (dataset, label, test_point):
  euclidean_distance = np.linalg.norm(dataset - test_point, axis = 1)
  
  target = np.argmin(euclidean_distance)
  print(target)
  return label[target]
  

dataset = np.array([[1,2],[3,1],[1,1]])
print(dataset.shape)
label = ['1', '2', '1']
test_point = np.array([1,2])

predict = predict_1nn(dataset, label, test_point)
print("predicted label: ", predict)