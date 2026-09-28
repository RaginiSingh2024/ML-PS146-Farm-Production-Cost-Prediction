"""
src/train_models.py
Script to train, evaluate, compare all 5 regression algorithms, and save the best model.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    load_data,
    get_feature_target_split,
    get_train_test_split,
    get_preprocessor,
    FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES
)
from src.evaluate_models import (
    calculate_metrics,
    plot_model_comparison,
    plot_actual_vs_predicted,
    plot_residual_distribution,
    plot_feature_importance,
    plot_farm_area_vs_cost
)

def train_and_evaluate_all():
    """
    Train and evaluate all 5 required regression models:
    1. Linear Regression
    2. Polynomial Regression
    3. Decision Tree Regression
    4. Random Forest Regression
    5. Gradient Boosting Regression
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # 1. Load data
    df = load_data("dataset/farm_production_cost.csv")
    X, y = get_feature_target_split(df)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Dataset split: Train = {X_train.shape[0]} records, Test = {X_test.shape[0]} records")
    
    # Preprocessors
    scaled_preprocessor = get_preprocessor(scale_numeric=True)
    unscaled_preprocessor = get_preprocessor(scale_numeric=False)
    
    # Define models dictionary
    models = {
        'Linear Regression': Pipeline([
            ('preprocessor', scaled_preprocessor),
            ('regressor', LinearRegression())
        ]),
        'Polynomial Regression': Pipeline([
            ('preprocessor', scaled_preprocessor),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('regressor', Ridge(alpha=1.0))
        ]),
        'Decision Tree Regression': Pipeline([
            ('preprocessor', unscaled_preprocessor),
            ('regressor', DecisionTreeRegressor(max_depth=6, min_samples_split=10, min_samples_leaf=5, random_state=42))
        ]),
        'Random Forest Regression': Pipeline([
            ('preprocessor', unscaled_preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=120, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42))
        ]),
        'Gradient Boosting Regression': Pipeline([
            ('preprocessor', unscaled_preprocessor),
            ('regressor', GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42))
        ])
    }
    
    results = []
    trained_pipelines = {}
    test_predictions = {}
    
    print("\n" + "="*70)
    print("TRAINING AND EVALUATING ALL 5 REGRESSION MODELS")
    print("="*70)
    
    for name, pipeline in models.items():
        print(f"\nTraining [{name}]...")
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline
        
        y_pred = pipeline.predict(X_test)
        test_predictions[name] = y_pred
        
        metrics = calculate_metrics(y_test, y_pred, model_name=name)
        results.append(metrics)
        print(f" -> MAE: ₹{metrics['MAE']:,.2f} | MSE: {metrics['MSE']:,.2f} | RMSE: ₹{metrics['RMSE']:,.2f} | R²: {metrics['R²']:.4f}")
        
    comparison_df = pd.DataFrame(results)
    
    # Sort models by RMSE ascending (lower is better) and R² descending (higher is better)
    comparison_df = comparison_df.sort_values(by=['RMSE', 'R²'], ascending=[True, False]).reset_index(drop=True)
    comparison_df.to_csv("results/model_comparison.csv", index=False)
    
    print("\n" + "="*70)
    print("FINAL MODEL COMPARISON TABLE (ACTUAL EXECUTION METRICS)")
    print("="*70)
    print(comparison_df.to_string(index=False))
    
    # Determine the best model objectively
    best_model_name = comparison_df.iloc[0]['Model']
    best_pipeline = trained_pipelines[best_model_name]
    best_y_pred = test_predictions[best_model_name]
    
    print(f"\n[BEST MODEL SELECTED]: {best_model_name}")
    print(f"Reason: Lowest RMSE (₹{comparison_df.iloc[0]['RMSE']:,.2f}) and Highest R² ({comparison_df.iloc[0]['R²']:.4f})")
    
    # Save best model pipeline
    best_model_path = "models/best_model.pkl"
    joblib.dump(best_pipeline, best_model_path)
    print(f"[SUCCESS] Saved best model pipeline to '{best_model_path}'.")
    
    # 2. Feature Importance Analysis
    # Use Gradient Boosting or Random Forest feature importances
    rf_pipeline = trained_pipelines['Random Forest Regression']
    gb_pipeline = trained_pipelines['Gradient Boosting Regression']
    
    # Get feature names after preprocessing
    proc = rf_pipeline.named_steps['preprocessor']
    feature_names = proc.get_feature_names_out()
    # Clean up feature names for presentation
    clean_feature_names = [f.replace('num__', '').replace('cat__', '') for f in feature_names]
    
    # Let's extract tree importances from Random Forest & Gradient Boosting
    rf_importances = rf_pipeline.named_steps['regressor'].feature_importances_
    gb_importances = gb_pipeline.named_steps['regressor'].feature_importances_
    
    # Aggregate importance per original feature category for clarity
    raw_importance_df = pd.DataFrame({
        'Transformed_Feature': clean_feature_names,
        'RF_Importance': rf_importances,
        'GB_Importance': gb_importances
    })
    
    # Aggregate crop one-hot columns into 'Crop_Type' and map back to the 9 primary features
    importance_map = {}
    for idx, row in raw_importance_df.iterrows():
        feat = row['Transformed_Feature']
        score = row['GB_Importance']
        if feat.startswith('Crop_Type'):
            importance_map['Crop_Type'] = importance_map.get('Crop_Type', 0.0) + score
        else:
            importance_map[feat] = score
            
    importance_list = [{'Feature': k, 'Importance': round(v, 4)} for k, v in importance_map.items()]
    importance_df = pd.DataFrame(importance_list).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    importance_df.to_csv("results/feature_importance.csv", index=False)
    
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE TABLE (NORMALIZED)")
    print("="*70)
    print(importance_df.to_string(index=False))
    print(f"\nMost influential feature identified: {importance_df.iloc[0]['Feature']} ({importance_df.iloc[0]['Importance']*100:.2f}%)")
    
    # 3. Generate visualizations
    plot_model_comparison(comparison_df, output_path="results/model_comparison.png")
    plot_actual_vs_predicted(y_test, best_y_pred, best_model_name, output_path="results/actual_vs_predicted.png")
    plot_residual_distribution(y_test, best_y_pred, best_model_name, output_path="results/residual_plot.png")
    plot_feature_importance(importance_df, output_path="results/feature_importance.png")
    plot_farm_area_vs_cost(df, output_path="results/farm_area_vs_cost.png")
    
    print("\n[SUCCESS] All models trained, evaluated, saved, and all charts generated in results/ directory!")
    return comparison_df, importance_df

if __name__ == "__main__":
    train_and_evaluate_all()
