import pandas as pd
import os
from sklearn.model_selection import train_test_split


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
   
  # 2. PREPROCESSING
  target_name = 'default payment next month'
    
  if 'ID' in df.columns:
    df = df.drop('ID', axis=1)
        
  X = df.drop(target_name, axis=1).values
  y = df[target_name].values

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2802, stratify=y)
  print(f"Training on {len(X_train)} samples...")
  print(f"Testing on {len(X_test)} samples...\n")