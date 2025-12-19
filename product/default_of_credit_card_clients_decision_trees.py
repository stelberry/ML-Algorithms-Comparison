import pandas as pd
import os
from sklearn.model_selection import train_test_split
from decision_trees import DecisionTreesCART
import numpy as np
from sklearn.utils import resample
import time


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

1. Limit Balance/Credit Limit: LIMIT_BAL
   - Amount of give credit in NT dollars

2. Demographics (4 features):
   - SEX: Gender (1=male, 2=female)
   - EDUCATION: Education level (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
   - MARRIAGE: Marital status (1=married, 2=single, 3=others)
   - AGE: Age in years

3. Repayment Status (6 features): PAY_0 to PAY_6
   - Repayment status from April to September 2005
   - These columns indicate if a customer paid on time or delayed payment for that specific month.
   - PAY_0: September....PAY_6: April
   - -2 = didn't use the card so nothing to pay (no consumption)
   - -1 = paid the entire balance on time
   -  0 = paid the minimum amount required but not the full balance. Not overdue but carrying a balance forward.
   -  1 = one month delay....8 = eight months delay
   
4. Bill Amounts (6 features): BILL_AMT1 to BILL_AMT6
   - How much money was billed to the customer in that month
   - Amount of bill statement from April to September 2005 (NT dollars)
   
5. Previous Payment Amount (6 features): PAY_AMT1 to PAY_AMT6
   - how much the customer actually paid in that month to settle their previous bill
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
    
  target_name = 'default payment next month'
  
  
  """
  ==========================================================
  Renaming Column
  The dataset has 'PAY_0' but then skips to 'PAY_2'.
  Rename 'PAY_0' -> 'PAY_1' to be consistent.
  ==========================================================
  """
  print("\nColumns before rename:")
  print(df.columns.tolist())
  df.columns = df.columns.str.strip()
  
  print("\n--- Renaming Column ---")
  df.rename(columns={'PAY_0': 'PAY_1'}, inplace=True)
  print("\nColumns after rename:")
  print(df.columns.tolist())
  print("\n===========================================")


  """
  ==========================================================
  Data Cleaning (Handling Undocumented Labels)
  Education: 0, 5, 6 are unlabelled. Group them into 4 (Others)
  Marriage: 0 is unlabelled. Group it into 3 (Others)
  ==========================================================
  """
  print("\n--- Cleaning Undocumented Labels ---")
  
  # check counts before cleaning
  print("Education values before:", np.sort(df['EDUCATION'].unique()))
  print("Marriage values before:", np.sort(df['MARRIAGE'].unique()))
  
  df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
  df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})
  
  print("\nEducation values after:", np.sort(df['EDUCATION'].unique()))
  print("Marriage values after:", np.sort(df['MARRIAGE'].unique()))
  print('\n========================================')
  
  """cat_features = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
  
  for col in cat_features:
    df[col] = df[col].astype(str)"""
      
  if 'ID' in df.columns:
    df = df.drop('ID', axis=1)
        
  X = df.drop(target_name, axis=1).values
  y = df[target_name].values
  
  # training a custom Python decision tree on 30,000 rows is very slow.
  # I sample 2,000 rows for development.
  """print("2,000 samples for speed...")
  X, y = resample(X, y, n_samples=3000, random_state=2802, stratify=y)"""


  """
  ==========================================================
  Class Imbalance Check
  This calculates and prints the exact % of Default vs Non-Default
  ==========================================================
  """
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
  print(f"\nTraining on {len(X_train)} samples...")
  print(f"Testing on {len(X_test)} samples...\n")
  
  criterions = ['gini', 'entropy']
  
  for criterion in criterions:
    print("=" * 50)
    print(f"TESTING WITH {criterion.upper()}")
    print("=" * 50)
    
    my_tree = DecisionTreesCART(max_depth=5, min_samples=2, criterion=criterion)
    
    print("Building the tree (this may take a moment)...")

    # ------------------------------
    # Training time (Decision Tree)
    # ------------------------------
    t_train_start = time.perf_counter()
    my_tree.fit(X_train, y_train)
    t_train_end = time.perf_counter()
    train_time = t_train_end - t_train_start
    
    """
    Visualize the tree
    """
    print("\n" + "="*30)
    print("DECISION TREE STRUCTURE")
    print("="*30)
    
    # get the column names so it says "PAY_1" instead of "Feature 5"
    feature_list = df.drop(target_name, axis=1).columns.tolist()
    
    my_tree.print_tree(my_tree.root, feature_names=feature_list)
    print("="*30 + "\n")
        
    print("Making predictions....")
    
    # ------------------------------
    # Prediction time (Decision Tree)
    # ------------------------------
    t_pred_start = time.perf_counter()
    predictions = my_tree.predict(X_test)
    t_pred_end = time.perf_counter()
    pred_time = t_pred_end - t_pred_start
    
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
    
    print("\n---------------- RUNTIME ----------------")
    print(f"[Runtime] Training time (fit): {train_time:.4f} seconds")
    print(f"[Runtime] Prediction time (full test set): {pred_time:.6f} seconds")
    print(f"[Runtime] Avg per test sample: {pred_time/len(X_test):.8f} seconds")
  
  print()
  
if __name__ == "__main__":
  run_credit_card_tree()