"""
Complete ML Pipeline for Heart Disease Classification
Combines preprocessing + model training in one workflow
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd

def create_pipeline():
    """Create the complete ML pipeline"""
    pipeline = Pipeline([
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    return pipeline

def train_pipeline(X_train, y_train):
    """Train complete pipeline"""
    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline

def save_pipeline(pipeline, filepath='models/final_pipeline.pkl'):
    """Save trained pipeline"""
    joblib.dump(pipeline, filepath)
    print(f"Pipeline saved to {filepath}")

def load_pipeline(filepath='models/final_pipeline.pkl'):
    """Load saved pipeline"""
    return joblib.load(filepath)

def predict_new_data(pipeline, new_data):
    """Make predictions on new data"""
    predictions = pipeline.predict(new_data)
    probabilities = pipeline.predict_proba(new_data)
    return predictions, probabilities

if __name__ == "__main__":
    from data_processing import load_data, handle_missing_values, preprocess_features
    
    # Process data
    df = load_data('data/dataset.csv')
    df = handle_missing_values(df)
    df = preprocess_features(df)
    
    # Prepare
    X = df.drop(['num', 'target'], axis=1, errors='ignore')
    y = df['target']
    
    # Train pipeline
    pipeline = train_pipeline(X, y)
    
    # Save
    save_pipeline(pipeline)
    
    print("Pipeline training complete!")
