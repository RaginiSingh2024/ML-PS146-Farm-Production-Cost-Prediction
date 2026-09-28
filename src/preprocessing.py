"""
src/preprocessing.py
Data loading, validation, feature separation, and sklearn preprocessing pipeline construction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Numerical and categorical feature definitions
CATEGORICAL_FEATURES = ['Crop_Type']
NUMERICAL_FEATURES = [
    'Farm_Area',
    'Seed_Cost',
    'Fertilizer_Usage',
    'Labor_Requirements',
    'Irrigation_Cost',
    'Pesticide_Usage',
    'Machinery_Cost',
    'Transportation_Cost'
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGET_COLUMN = 'Total_Production_Cost'

def load_data(filepath: str = "dataset/farm_production_cost.csv") -> pd.DataFrame:
    """Load dataset from CSV file."""
    df = pd.read_csv(filepath)
    return df

def get_feature_target_split(df: pd.DataFrame):
    """
    Separate features (X) and target variable (y).
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y

def get_train_test_split(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Split feature matrix and target vector into training and testing subsets
    with a reproducible random_state.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def get_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer that encodes categorical features
    and scales numerical features.
    
    This ensures that preprocessing is tightly coupled with the model in a single
    reusable pipeline, completely preventing data leakage between train and test sets.
    """
    transformers = [
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
    ]
    
    if scale_numeric:
        transformers.append(('num', StandardScaler(), NUMERICAL_FEATURES))
    else:
        transformers.append(('num', 'passthrough', NUMERICAL_FEATURES))
        
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )
    return preprocessor

if __name__ == "__main__":
    df = load_data()
    print(f"Loaded dataset with shape {df.shape}")
    X, y = get_feature_target_split(df)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    preprocessor = get_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    print(f"Preprocessed X_train shape: {X_train_proc.shape}")
