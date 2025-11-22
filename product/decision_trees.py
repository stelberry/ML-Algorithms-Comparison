import numpy as np

class Node: 
  """A helper class to store tree info instead of using confusing strings."""
  def __init__(self, feature = None, threhold = None, left = None, right = None, value = None):
    self.feature = feature
    self.threhold = threhold
    self.left = left
    self.right = right
    self.value = value
    
  def is_leaf(self):
    return self.value is not None
    
 
  






  def calc_entropy(labels):
    class_names, class_counts = np.unique(labels, return_count = True)
    probs = class_counts / class_counts.sum
    entropy = -np.sum(probs * np.log2(probs))
    return entropy
  
  def calc_gini_impurity(labels):
    class_names, class_counts = np.unique(labels, return_count = True)
    probs = class_counts/class_counts.sum
    gini_impurity = 1.0 - np.sum(probs ** 2)
    return gini_impurity



