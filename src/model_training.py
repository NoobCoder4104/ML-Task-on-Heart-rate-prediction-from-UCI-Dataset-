"""
Model Training Script
Trains Logistic Regression, Decision Tree, Random Forest and compares results
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json
from datetime import datetime

def split_data(X, y, test_size=0.2, random_state=42):
    """Split with stratification"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    """Train multiple models"""
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'DecisionTree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"✓ {name} trained")
    
    return trained

def evaluate_models(models, X_test, y_test):
    """Calculate all metrics"""
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
    
    return results

def save_model(model, filepath='models/pipeline.pkl'):
    """Save trained model"""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def save_predictions(y_test, predictions, filepath='outputs/predictions.csv'):
    """Save predictions to CSV"""
    df = pd.DataFrame({
        'actual': y_test,
        'logistic_regression': predictions.get('LogisticRegression'),
        'decision_tree': predictions.get('DecisionTree'),
        'random_forest': predictions.get('RandomForest')
    })
    df.to_csv(filepath, index=False)
    print(f"Predictions saved to {filepath}")

# Run if executed directly
if __name__ == "__main__":
    from data_processing import load_data, handle_missing_values, preprocess_features
    
    # Load and process
    df = load_data('data/dataset.csv')
    df = handle_missing_values(df)
    df = preprocess_features(df)
    
    # Prepare for training
    X = df.drop(['num', 'target'], axis=1, errors='ignore')
    y = df['target']
    
    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train
    models = train_models(X_train, y_train)
    
    # Evaluate
    results = evaluate_models(models, X_test, y_test)
    print("\nResults:", json.dumps(results, indent=2))
    
    # Save best model (Random Forest)
    save_model(models['RandomForest'])
