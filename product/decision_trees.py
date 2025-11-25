import numpy as np
from collections import Counter


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
    
  def find_best_split(self, features, labels, n_features):
    """Loops through all features to find the split with the lowest Gini."""
    
    best_gini = 1.0
    best_split = None
    
    for feature_index in range(n_features):
      current_column_values = features[:, feature_index]
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
          best_split = {'feature_index': feature_index,
                        'threshold': threshold,
                        'left_features': features[left_mask],  
                        'left_labels': left_targets,               
                        'right_features': features[right_mask],
                        'right_labels': right_targets
                        }
    return best_split
       
  def create_tree(self,features, labels, depth=0):
    n_samples, n_features = features.shape
    n_unique_labels = len(np.unique(labels))
        
    #Stopping criteria
    if (depth >= self.max_depth) or (n_samples < self.min_samples) or (n_unique_labels == 1):
      most_common_label = Counter(labels).most_common(1)[0][0]
      return Node(value = most_common_label)
      
    best_split = self.find_best_split(features, labels, n_features)
    
    if best_split is None:
      most_common_label = Counter(labels).most_common(1)[0][0]
      return Node(value = most_common_label)
    
    left_child = self.create_tree(best_split['left_features'], best_split['left_labels'], depth + 1)
    right_child = self.create_tree(best_split['right_features'], best_split['right_labels'], depth + 1)
    
    return Node(feature = best_split['features_index'],
                threshold = best_split['threshold'],
                left = left_child,
                right = right_child
                )
      
  def fit(self, features, labels):
    self.root = self.create_tree(np.array(features), np.array(labels))
      
    