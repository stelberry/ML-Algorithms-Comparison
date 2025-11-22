import numpy as np

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

