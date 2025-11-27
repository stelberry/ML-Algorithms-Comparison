import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from one_NN import predict_1nn
import os

def credit_card_1nn():
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
    df = df.drop('ID', axis = 1)
  
  X = df.drop(target_name, axis =1).values
  y = df[target_name].values
  
  feature_names = df.drop(target_name, axis=1).columns
  
  print("Original Dataset Shape:", X.shape)
  
  # since 30,000 rows is too slow for a simple 1NN loop, I sample 1,000 rows for testing.
  # stratify=y to keep the same % of default/non-default.
  X, y = resample(X, y, n_samples=1000, random_state=2802, stratify=y)
  print("Speeded Sampled Shape:", X.shape)
  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    
  print(f"\nTraining data shape: {X_train.shape}")
  print(f"Testing data shape: {X_test.shape}")
    
  print("\nTraining 1NN model...")
  
  evaluation_arr = []
  print("Starting predictions (this might take a moment)...")
  
  for test_point in X_test:
    predict = predict_1nn(X_train, y_train, test_point)
    evaluation_arr.append(predict)
    
  y_pred = np.array(evaluation_arr)
  
  accuracy_score = np.mean(y_pred == y_test)
  error_rate = 1 - accuracy_score
  
  print("Predicted labels (first 10): ", y_pred[:10])
  print("Actual labels (first 10):    ", y_test[:10])
  print("Accuracy score:", accuracy_score)
  print("Error rate:",error_rate)
  
  
  
if __name__ == "__main__":
  credit_card_1nn()
  