"""
Train a Random Forest model to predict house prices.
Generates synthetic data, trains the model, and saves artifacts.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_data(n_samples=1000):
    """
    Generate synthetic house price dataset with 7 features.
    
    Returns:
        pd.DataFrame: Dataset with features and target price
    """
    data = {
        'square_feet': np.random.randint(800, 4000, n_samples),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.randint(1, 5, n_samples),
        'age_years': np.random.randint(0, 50, n_samples),
        'lot_size': np.random.randint(2000, 10000, n_samples),
        'garage_spaces': np.random.randint(0, 4, n_samples),
        'neighborhood_score': np.random.uniform(1, 10, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Generate realistic price based on features with some noise
    price = (
        df['square_feet'] * 150 +
        df['bedrooms'] * 10000 +
        df['bathrooms'] * 15000 +
        df['age_years'] * (-1000) +
        df['lot_size'] * 20 +
        df['garage_spaces'] * 8000 +
        df['neighborhood_score'] * 25000 +
        np.random.normal(0, 30000, n_samples)  # Add noise
    )
    
    df['price'] = price
    
    return df

def train_model():
    """
    Train Random Forest model and save artifacts.
    """
    print("Generating synthetic dataset...")
    df = generate_synthetic_data(1000)
    
    # Save dataset
    df.to_csv('data/house_prices.csv', index=False)
    print(f"Dataset saved to data/house_prices.csv")
    print(f"Dataset shape: {df.shape}")
    print(f"\nDataset statistics:")
    print(df.describe())
    
    # Prepare features and target
    feature_names = ['square_feet', 'bedrooms', 'bathrooms', 'age_years', 
                     'lot_size', 'garage_spaces', 'neighborhood_score']
    X = df[feature_names]
    y = df['price']
    
    # Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Normalize features with StandardScaler
    print("\nNormalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest Regressor
    print("\nTraining Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    print(f"\nTraining Results:")
    print(f"  MAE: ${train_mae:,.2f}")
    print(f"  R² Score: {train_r2:.4f}")
    
    print(f"\nTest Results:")
    print(f"  MAE: ${test_mae:,.2f}")
    print(f"  R² Score: {test_r2:.4f}")
    
    # Save model artifacts
    print("\nSaving model artifacts...")
    
    # Save to root directory for HuggingFace deployment
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    print("Model saved to: model.pkl")
    print("Scaler saved to: scaler.pkl")
    print("Feature names saved to: feature_names.pkl")
    
    # Also save to models directory for local use
    os.makedirs('models', exist_ok=True)
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    print("\nModel training complete!")

if __name__ == "__main__":
    train_model()
