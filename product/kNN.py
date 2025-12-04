import numpy as np
from collections import Counter

def predict_knn(dataset, labels, test_point, k):
  """
  Compute the Euclidean distance between the test_point and each row in the dataset
  For example, datasets are [1,2] and [3,1] and test_point is [4,2].
  
  1st: (dataset- testpoint),
  Row 0: [1,2] - [4,2] = [-3,0]
  Row 1: [3,1] - [4,2] = [-1,-1]
  
  2nd: np.linalg.norm numpy function compute the norm length(squared each then sum all then sqrt that sum)
  """
  euclidean_distance = np.linalg.norm(dataset - test_point, axis = 1)
  #print("distances:", euclidean_distance[:10])
  
  # get the indices that would sort the distances from smallest to largest
  sort_indicies = np.argsort(euclidean_distance)
  k_indicies = sort_indicies[:k]
  #print("k indicies:", k_indicies)
  
  # get the actual labels (e.g. 'A', 'B'/ '0', '1') for the k-closest neighbors
  # this list is automatically sorted by distance (closest first)
  k_nearest_labels = labels[k_indicies]
  #print("k nearest labels:", k_nearest_labels)
  
  """# randomly selecting label
  #this just picks one random neighbor from the k-list. It doesn't vote.
  prediction = np.random.choice(k_nearest_labels)
  return prediction"""
  
  """# choose alphabetically/numerical priority 
  #this counts votes, but breaks ties alphabetically/numerical priority (e.g., 'A' wins 'B'/smallest number wins).
  unique_labels, counts = np.unique(k_nearest_labels, return_counts = True)
  max_count_index = np.argmax(counts)
  prediction = unique_labels[max_count_index]
  return prediction"""
  
  # choose by distance
  # this counts votes. In a tie, it picks the neighbor that was closest.
  # 1. Counter() counts votes (e.g., {'C': 2, 'A': 2, 'B': 1})
  #    it remembers the order it saw them (C was closer than A).
  
  # 2. .most_common(1) picks the one it saw first.
  # since k_nearest_labels is sorted by distance, this naturally breaks ties by choosing the closest neighbor
  most_common = Counter(k_nearest_labels).most_common(1)
  
  #if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
    #print(f"TIE DETECTED! k={k}, labels={k_nearest_labels}, winner={most_common[0][0]}")
  
  # get the winning label (e.g., 'C') from the list [('C', 2)]
  prediction = most_common[0][0]
  return prediction
  
"""
 #test data
data = np.array([[1,2],[2,2],[4,3],[3,1],[5,2],[1,1],[2,3]])
label = np.array(['A','B','A','C','B','A','C'])
test_point = [3,2]

# run the prediction using k=5
prediction = predict_knn(data, label, test_point, k=5)
print("Predicted label:", prediction)
"""