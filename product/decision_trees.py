import numpy as np

class Node: 
  """A helper class to store tree info instead of using confusing strings."""
  def __init__(self, feature = None, threshold = None, left = None, right = None, value = None):
    self.feature = feature
    self.threshold = threshold
    self.left = left
    self.right = right
    self.value = value
    
  def is_leaf(self):
    return self.value is not None
    
class DecisionTreesCART:
  def __init__(self, max_depth=10, min_samples=2):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root = None
        
  def calc_entropy(self, labels):
    class_names, class_counts = np.unique(labels, return_counts = True)
    probs = class_counts / class_counts.sum()
    entropy = -np.sum(probs * np.log2(probs))
    return entropy
  
  def calc_gini_impurity(self, labels):
    class_names, class_counts = np.unique(labels, return_counts = True)
    probabilities = class_counts/class_counts.sum()
    gini_impurity = 1.0 - np.sum(probabilities ** 2)
    return gini_impurity

  def calc_weighted_gini(self, left_labels, right_labels):
    total = len(left_labels) + len(right_labels)
    
    gini_left = self.calc_gini_impurity(left_labels)
    gini_right = self.calc_gini_impurity(right_labels)
    
    left_weight = len(left_labels)/total
    right_weight = len(right_labels)/total
    
    weighted_gini = (left_weight * gini_left) + (right_weight * gini_right)
  
    return weighted_gini
    
  def find_best_split(self, features, labels, num_features):
    """Loops through all features to find the split with the lowest Gini."""
    
    best_gini = 1.0
    best_split = None
    
    for features_index in range(num_features):
      current_column_values = features[:, features_index]
      thresholds = np.unique(current_column_values)
      
      for threshold in thresholds:
      
        if isinstance(threshold, str):
          left_mask = current_column_values == threshold
          right_mask = current_column_values != threshold

        else:
          left_mask = current_column_values <= threshold
          right_mask = current_column_values > threshold
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
          continue
          
        left_targets = labels[left_mask]
        right_targets = labels[right_mask]
        
        gini = self.calc_weighted_gini(left_targets, right_targets)
        
        if gini < best_gini:
          best_gini = gini
          best_split = {'feature_index': features_index,
                        'threshold': threshold,
                        }
    return best_split
        
        


