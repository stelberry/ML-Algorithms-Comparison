import pandas as pd
import os
from sklearn.model_selection import train_test_split
from decision_trees import DecisionTreesCART
import numpy as np
from sklearn.utils import resample


"""
UCI Default of Credit Cart Clients Dataset
================================

DATASET BACKGROUND:
This dataset addresses customer default payments in Taiwan, where card-issuing 
banks over-issued cash and credit cards to unqualified applicants to increase 
market share, while cardholders overused credit cards and accumulated heavy debts .

TARGET VARIABLE:
- default payment next month: Binary (1 = yes, default; 0 = no, not default)
  * Predicting whether a client will fail to make their payment next month

FEATURES (23 total):

1. Demographics (4 features):
   - LIMIT_BAL: Credit amount in NT dollars (includes individual and family credit)
   - SEX: Gender (1=male, 2=female)
   - EDUCATION: Education level (1=graduate school, 2=university, 3=high school, 4=others)
   - MARRIAGE: Marital status (1=married, 2=single, 3=others)
   - AGE: Age in years

2. Payment History (6 features): PAY_0 to PAY_6
   - Repayment status from April to September 2005
   - Values indicate months of payment delay
   - -1,-2,0 = pay on time, 1 = one month delay, 2 = two months delay, etc.

3. Bill Amounts (6 features): BILL_AMT1 to BILL_AMT6
   - Amount of bill statement from April to September 2005 (NT dollars)
   
4. Previous Payments (6 features): PAY_AMT1 to PAY_AMT6
   - Amount of previous payment from April to September 2005 (NT dollars)


DATASET CHARACTERISTICS:
- Imbalanced Data (22% Default vs 78% Pay)
- Mix of categorical (sex, education, marriage) and continuous features
- Feature scales vary widely (age: 20-80, credit limit: 10,000-1,000,000)
"""
def run_credit_card_tree():
  filename = 'product/default_of_credit_card_clients.xls'
    
  print(f"Loading '{filename}'... (this might take a few seconds)")
    
  try:
    df = pd.read_excel(filename, header=1)
  except FileNotFoundError:
    print(f"\nERROR: Could not find '{filename}'.")
    print(f"Current folder: {os.getcwd()}")
    return
    
  #Preprocessing
  target_name = 'default payment next month'
    
  if 'ID' in df.columns:
    df = df.drop('ID', axis=1)
        
  X = df.drop(target_name, axis=1).values
  y = df[target_name].values
  
  """# training a custom Python decision tree on 30,000 rows is very slow.
  # I sample 2,000 rows for development.
  print("2,000 samples for speed...")
  X, y = resample(X, y, n_samples=2000, random_state=2802, stratify=y)"""


  """==========================================================
  # NEW: Class Imbalance Check
  # This calculates and prints the exact % of Default vs Non-Default
  # =========================================================="""
  unique, counts = np.unique(y, return_counts=True)
  total_samples = len(y)

  print("\n--- Class Imbalance Analysis ---")
  
  for i in range(len(unique)):
      cls_label = unique[i]    # [0,1]
      count = counts[i]        # [23364,6636]
      percent = (count / total_samples) * 100
      
      if cls_label == 1:
          print(f"Default (1):     {count} samples ({percent}%)")
      else:
          print(f"Non-Default (0): {count} samples ({percent}%)")
          
          
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2802, stratify=y)
  print(f"Training on {len(X_train)} samples...")
  print(f"Testing on {len(X_test)} samples...\n")
  
  criterions = ['gini', 'entropy']
  
  for criterion in criterions:
    print("=" * 50)
    print(f"TESTING WITH {criterion.upper()}")
    print("=" * 50)
    
    my_tree = DecisionTreesCART(max_depth=5, min_samples=2, criterion=criterion)
    
    print("Building the tree (this may take a moment)...")
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
    labels = ['No Default', 'Default']
    for i in range(10):
      actual = labels[y_test[i]]
      predicted = labels[predictions[i]]
      print(f"Actual: {actual}, Predicted: {predicted}")
  print()
  
if __name__ == "__main__":
  run_credit_card_tree()