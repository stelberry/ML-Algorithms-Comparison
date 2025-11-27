import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from one_NN import predict_1nn

def credit_card_1nn():
  try:
    df = pd.read_csv('defalut of credit card clients.csv')
  except FileNotFoundError:
    print("csv file not found")
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
  y = df[target_name].value
  
  feature_names = df.drop(target_name, axis=1).columns
  
  print("Original Dataset Shape:", X.shape)
  
  # since 30,000 rows is too slow for a simple 1NN loop, I sample 1,000 rows for testing.
  # stratify=y to keep the same % of default/non-default.
  X, y = resample(X, y, n_samples=1000, random_state=42, stratify=y)
  print("Speeded Sampled Shape:", X.shape)
  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    
  print(f"\nTraining data shape: {X_train.shape}")
  print(f"Testing data shape: {X_test.shape}")
    
  print("\nTraining 1NN model...")
  print("This may take a while for large datasets...")