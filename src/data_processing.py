"""
Data Processing Script for Heart Disease Dataset
Handles: Loading, cleaning, missing values, feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Load raw dataset"""
    df = pd.read_csv(filepath, na_values='?')
    print(f"Data loaded: {df.shape}")
    return df

def handle_missing_values(df):
    """Drop high-missing columns and impute others"""
    # Drop columns with >50% missing
    high_missing = df.columns[df.isnull().mean() > 0.5].tolist()
    print(f"Dropping columns: {high_missing}")
    df = df.drop(columns=high_missing)
    
    # Separate numeric and categorical
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    
    # Remove target from imputation
    for target in ['num', 'target']:
        if target in num_cols:
            num_cols.remove(target)
    
    # Impute numeric with median
    num_imputer = SimpleImputer(strategy='median')
    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    
    # Impute categorical with mode
    if cat_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    
    print(f"Missing values remaining: {df.isnull().sum().sum()}")
    return df

def preprocess_features(df):
    """Feature engineering and scaling"""
    # Create target if needed
    if 'num' in df.columns:
        df['target'] = (df['num'] > 0).astype(int)
    
    # Encode categorical
    cat_cols = [col for col in ['cp', 'restecg'] if col in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Create new features
    df['chol_age_ratio'] = df['chol'] / df['age']
    df['thalach_ratio'] = df['thalach'] / (220 - df['age'])
    
    # Scale numeric features
    scale_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 
                  'chol_age_ratio', 'thalach_ratio']
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    
    print(f"Features prepared: {df.shape}")
    return df

# Run if executed directly
if __name__ == "__main__":
    df = load_data('data/dataset.csv')
    df = handle_missing_values(df)
    df = preprocess_features(df)
    print("Processing complete!")
