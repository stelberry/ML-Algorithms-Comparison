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
    
class DecisionTreesCart:

  def calc_entropy(labels):
    class_names, class_counts = np.unique(labels, return_count = True)
    probs = class_counts / class_counts.sum
    entropy = -np.sum(probs * np.log2(probs))
    return entropy
  
  def calc_gini_impurity(self, labels):
    class_names, class_counts = np.unique(labels, return_count = True)
    probabilities = class_counts/class_counts.sum
    gini_impurity = 1.0 - np.sum(probabilities ** 2)
    return gini_impurity

  def clac_weighted_gini(self, left_labels, right_labels):
    total = left_labels.sum + right_labels.sum
    
    gini_left = self.calc_gini_impurity(left_labels)
    gini_right = self.calc_gini_impurity(right_labels)
    
    left_weight = left_labels.sum/total
    right_weight = right_labels.sum/total
    
    weighted_gini = (left_weight * gini_left) + (right_weight * gini_right)
  
    return weighted_gini
    
  


