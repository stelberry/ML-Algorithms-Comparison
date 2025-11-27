import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from decision_trees import DecisionTreesCART

"""
The dataset consists of 3 types of Iris flowers. 
We measure 4 for every 150 flower: Sepal Length, Sepal Width, Petal Length, and Petal Width.
X (data): measurements (e.g., 5.1, 3.5, 1.4, 0.2)
y (target): species name in int
0 = Setosa
1 = Versicolor
2 = Virginica
"""

def run_iris_test():

  print("Loading Iris Dataset...")
  iris = load_iris()
  X = iris['data']
  y = iris['target']

  X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 2802)
  
  print(f"Training on {len(X_train)} samples...")
  print(f"Testing on {len(X_test)} samples...\n")
  
  criterions = ['gini', 'entropy']

  for criterion in criterions:
    print("=" * 50)
    print(f"TESTING WITH {criterion.upper()}")
    print("=" * 50)
        
    my_tree = DecisionTreesCART(max_depth = 4, min_samples = 2, criterion = criterion)
    
    print("Building the tree...")
    my_tree.fit(X_train, y_train)
    
    print("Making predictions....")
    predictions = my_tree.predict(X_test)
    
    y_pred = np.array(predictions)
    accuracy_score = np.mean(y_pred==y_test) 
    total_guesses = len(y_test)
    correct_guesses = np.sum(y_pred == y_test)
    
    print("-" * 30)
    print(f"FINAL ACCURACY: {accuracy_score * 100:.2f}%")
    print(f"Correct: {correct_guesses}/{total_guesses}")
    print("-" * 30)
    
    print("\nSample Predictions:")
    flower_names = iris.target_names
    for i in range(5):
      actual_name = flower_names[y_test[i]]
      predicted_name = flower_names[predictions[i]]
      print(f"Actual: {actual_name}, Predicted: {predicted_name}")
  
  print()
  
if __name__ == "__main__":
      run_iris_test()