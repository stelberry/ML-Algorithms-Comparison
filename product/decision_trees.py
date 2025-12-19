import numpy as np
from collections import Counter

class Node: 
  def __init__(self, feature = None, threshold = None, 
              left = None, right = None, value = None):
              
    # the column index used to split the data
    self.feature = feature
    
    # the value used to split (e.g., if age > 25)
    self.threshold = threshold
    
    # the node where data goes if the condition is true
    self.left = left
    
    # the node where data goes if the condition is false
    self.right = right
    
    # if this is a final leaf node, this holds the prediction class
    self.value = value
    
  # helper to check the end of a branch
  def is_leaf(self):
    return self.value is not None
    
class DecisionTreesCART:
  def __init__(self, max_depth=10, min_samples=2, criterion = 'gini'):
  
        # stops the tree from growing too complex (overfitting)
        self.max_depth = max_depth
        
        # stops splitting if there aren't enough items
        self.min_samples = min_samples
      
        self.root = None
        
        # chooses between 'gini' or 'entropy' math
        self.criterion = criterion
        
  def calc_entropy(self, labels):
    # counts how many of each class there are
    class_names, class_counts = np.unique(labels, return_counts = True)
    probs = class_counts / class_counts.sum()
    probs = probs[probs > 0] # only take logs of probabilities > 0
    entropy = -np.sum(probs * np.log2(probs))
    return entropy
  
  def calc_gini_impurity(self, labels):
    class_names, class_counts = np.unique(labels, return_counts = True)
    probabilities = class_counts/class_counts.sum()
    gini_impurity = 1.0 - np.sum(probabilities ** 2)
    return gini_impurity

  def calc_weighted_gini(self, left_labels, right_labels):
    # total number of items in this split
    total = len(left_labels) + len(right_labels)
    
    # calculate score for both sides
    gini_left = self.calc_gini_impurity(left_labels)
    gini_right = self.calc_gini_impurity(right_labels)
    
    left_weight = len(left_labels)/total
    right_weight = len(right_labels)/total
    
    # combine scores based on size
    weighted_gini = (left_weight * gini_left) + (right_weight * gini_right)
  
    return weighted_gini
   
  def calc_weighted_entropy(self, left_labels, right_labels):
    total = len(left_labels) + len(right_labels)
    
    entropy_left = self.calc_entropy(left_labels)
    entropy_right = self.calc_entropy(right_labels)
    
    left_weight = len(left_labels)/total
    right_weight = len(right_labels)/total
    
    weighted_entropy = (left_weight * entropy_left) + (right_weight * entropy_right)
    
    return weighted_entropy
    
  def _create_split_masks(self, column_values, threshold):
        """Create boolean masks for splitting data."""
        # handles text data (categorical)
        if isinstance(threshold, str):
            left_mask = column_values == threshold
            right_mask = column_values != threshold
        else:
        # handles number data (numerical)
            left_mask = column_values <= threshold
            right_mask = column_values > threshold
        return left_mask, right_mask

  def find_best_split(self, features, labels, n_features):
    """Loops through all features(columns) to find the split with the lowest impurity."""
    
    best_impurity = float('inf') #start with inifinity
    best_split = None
    
    # loop over every column in the dataset
    for feature_index in range(n_features):
      current_column_values = features[:, feature_index]
      
      # get unique values to test as thresholds
      thresholds = np.unique(current_column_values)
      
      # loop over every unique value in this column
      for threshold in thresholds:
        left_mask, right_mask = self._create_split_masks(current_column_values, threshold)     
        
        # skip if one side is empty (useless split)
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
          continue
          
        left_labels = labels[left_mask]
        right_labels = labels[right_mask]
        
        # calculate impurity based on criterion
        if self.criterion == 'gini':
          impurity = self.calc_weighted_gini(left_labels, right_labels)
        else:  # entropy
          impurity = self.calc_weighted_entropy(left_labels, right_labels)
        
        # if this split is cleaner than previous ones, save it
        if impurity < best_impurity:
          best_impurity = impurity
          best_split = {'feature_index': feature_index,
                        'threshold': threshold,
                        'left_features': features[left_mask],  
                        'left_labels': left_labels,               
                        'right_features': features[right_mask],
                        'right_labels': right_labels
                        }
    return best_split
    
  # simple voting: returns the most frequent class label    
  def get_most_common_labels(self, labels):
    most_common_labels = Counter(labels).most_common(1)[0][0]
    return Node(value = most_common_labels)
    
  def create_tree(self,features, labels, depth=0):
    n_samples, n_features = features.shape
    n_unique_labels = len(np.unique(labels))
        
    # stopping criteria
    # checks if we should stop growing the tree
    if (depth >= self.max_depth or
        n_samples < self.min_samples or
        n_unique_labels == 1):
        return self.get_most_common_labels(labels)
    
    # find the best question to ask to split data
    best_split = self.find_best_split(features, labels, n_features)
    
    # if no split was found, return a leaf node
    if best_split is None:
      return self.get_most_common_labels(labels)
    
    # recursively build the left branch
    left_child = self.create_tree(best_split['left_features'], 
                                  best_split['left_labels'], depth + 1)
                                  
    # recursively build the right branch 
    right_child = self.create_tree(best_split['right_features'], 
                                  best_split['right_labels'], depth + 1)
                                  
    # return the node connecting these two branches
    return Node(feature = best_split['feature_index'],
                threshold = best_split['threshold'],
                left = left_child,right = right_child)
      
  def fit(self, features, labels):
    # starts the whole building process
    self.root = self.create_tree(np.array(features), np.array(labels))
      
  def _traverse_tree(self, features, node):
        """Recursively traverse tree to make prediction for a single sample."""
        # if we hit the bottom, return the prediction
        if node.is_leaf():
            return node.value
            
        # get the value for the feature used in this node
        feature_value = features[node.feature]
        
        # decide which way to go (left or right) 
        if isinstance(node.threshold, str):
            if feature_value == node.threshold:
                return self._traverse_tree(features, node.left)
            else:
                return self._traverse_tree(features, node.right)
        else:
            if feature_value <= node.threshold:
                return self._traverse_tree(features, node.left)
            else:
                return self._traverse_tree(features, node.right)
            
  def predict(self, features):
    """Make predictions for multiple samples."""
    X = np.array(features)
    # run the traverse function for every row in the data
    return np.array([self._traverse_tree(x, self.root) for x in X]) 
    
  def print_tree(self, node, depth=0, feature_names=None):
    """
    Recursively prints the decision tree structure.
    """
    indent = "  " * depth
    if node.is_leaf():
        print(f"{indent}--> Leaf: Class {node.value}")
        return

    feature_label = f"Feature_{node.feature}"
    if feature_names is not None:
        feature_label = feature_names[node.feature]
    
    operator = "<="
    if isinstance(node.threshold, str):
        operator = "=="
        
    print(f"{indent}[{feature_label} {operator} {node.threshold}]")
    
    print(f"{indent}  True Path:")
    self.print_tree(node.left, depth + 1, feature_names)
    
    print(f"{indent}  False Path:")
    self.print_tree(node.right, depth + 1, feature_names)
    
        