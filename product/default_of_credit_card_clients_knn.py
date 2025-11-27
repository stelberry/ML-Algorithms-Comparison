import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from kNN import predict_knn
import os
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler

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
   - -1 = pay duly, 0 = no delay, 1 = one month delay, 2 = two months delay, etc.

3. Bill Amounts (6 features): BILL_AMT1 to BILL_AMT6
   - Amount of bill statement from April to September 2005 (NT dollars)
   
4. Previous Payments (6 features): PAY_AMT1 to PAY_AMT6
   - Amount of previous payment from April to September 2005 (NT dollars)


DATASET CHARACTERISTICS:
- Imbalanced: ~22% default, ~78% no default
- Mix of categorical (sex, education, marriage) and continuous features
- Feature scales vary widely (age: 20-80, credit limit: 10,000-1,000,000)
"""

def credit_card_knn():
  filename = 'product/default_of_credit_card_clients.xls'
    
  print(f"Loading '{filename}'... (this might take a few seconds)")
  
  try:
    df = pd.read_excel(filename, header=1)
  except FileNotFoundError:
    print(f"\nERROR: Could not find '{filename}'.")
    print(f"Current folder: {os.getcwd()}")
    return
  
  print("Dataset shape:", df.shape)
  print("\nFirst few rows:")
  print(df.head())
  print("\nColumn names:")
  print(df.columns.tolist())
  print("\nDataset info:")
  print(df.info())
  
  target_name = 'default payment next month'
  
    # drop 'ID' column as it is not a feature

  if 'ID' in df.columns:
        df = df.drop('ID', axis=1)
    
  # separate features (X) and target (y)
  X = df.drop(target_name, axis=1).values
  y = df[target_name].values

  print("\nOriginal Dataset Shape:", X.shape)
  
  """
  # since 30,000 rows is too slow for a simple 1NN loop, I sample 1,000 rows for testing.
  X, y = resample(X, y, n_samples=1000, random_state=0, stratify=y)
"""
  # split data into training set (75%) and testing set (25%)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2802, stratify=y)
  
  """
  transform all features to be in range [0, 1]
  without scaling, features with larger values such as 
  credit limit: 10,000-1,000,000 dominate the distance calculation
  over small values like Age: 20-80
  """
  scaler = MinMaxScaler()
  X_train = scaler.fit_transform(X_train) # learn the scaling from training data
  X_test = scaler.transform(X_test) # apply same scaling to test data
  
  print(f"\nTraining data shape: {X_train.shape}")
  print(f"Testing data shape: {X_test.shape}")
  
  k_value = 5 
  print(f"\nStarting k-NN predictions with k={k_value}...")
  
  # create empty list to store all predictions
  evaluation_arr = []
  print("Starting predictions (this might take a moment)...")
  print()
  
  for test_point in X_test:
    predict = predict_knn(X_train, y_train, test_point, k_value)
    evaluation_arr.append(predict)
    
  y_pred = np.array(evaluation_arr)
  
  accuracy_score = np.mean(y_pred == y_test)
  error_rate = 1 - accuracy_score
  
  print("\n---------------- RESULTS ----------------")
  print("k value:", k_value)
  print("Predicted labels (first 10): ", y_pred[:10])
  print("Actual labels (first 10):    ", y_test[:10])
  print("Accuracy score:", accuracy_score)
  print("Error rate:",error_rate)



if __name__ == "__main__":
  credit_card_knn()