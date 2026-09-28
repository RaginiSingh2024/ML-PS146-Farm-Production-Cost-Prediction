"""
src/prediction.py
Reusable prediction module for estimating farm production costs using the trained best model.
"""

import os
import joblib
import pandas as pd
from typing import Union, Dict

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl")

# Cached model instance
_CACHED_MODEL = None

def load_best_model(model_path: str = DEFAULT_MODEL_PATH):
    """
    Load the trained best model pipeline from disk.
    Caches the loaded model in memory for fast inference.
    """
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model not found at '{model_path}'. "
                f"Please run 'python src/train_models.py' first."
            )
        _CACHED_MODEL = joblib.load(model_path)
    return _CACHED_MODEL

def predict_production_cost(
    crop_type: str,
    farm_area: float,
    seed_cost: float,
    fertilizer_usage: float,
    labor_requirements: float,
    irrigation_cost: float,
    pesticide_usage: float,
    machinery_cost: float,
    transportation_cost: float,
    model=None
) -> float:
    """
    Predict the total farm production cost for a new farming scenario.
    
    Parameters:
    -----------
    crop_type : str
        Type of crop (e.g., 'Wheat', 'Rice', 'Cotton', 'Sugarcane', 'Maize')
    farm_area : float
        Total farm land area in acres (> 0)
    seed_cost : float
        Expenditure on seeds in ₹ (>= 0)
    fertilizer_usage : float
        Amount of fertilizer used in kg (>= 0)
    labor_requirements : float
        Total labor person-days required (>= 0)
    irrigation_cost : float
        Total irrigation/pumping cost in ₹ (>= 0)
    pesticide_usage : float
        Total pesticide quantity in liters/kg (>= 0)
    machinery_cost : float
        Machinery, tractor, harvester costs in ₹ (>= 0)
    transportation_cost : float
        Transportation and logistics costs in ₹ (>= 0)
    model : optional
        Pre-loaded scikit-learn Pipeline. If None, loads from disk.
        
    Returns:
    --------
    float: Predicted Total Production Cost in ₹ rounded to 2 decimal places.
    """
    if model is None:
        model = load_best_model()
        
    input_data = pd.DataFrame([{
        'Crop_Type': str(crop_type).strip(),
        'Farm_Area': float(farm_area),
        'Seed_Cost': float(seed_cost),
        'Fertilizer_Usage': float(fertilizer_usage),
        'Labor_Requirements': float(labor_requirements),
        'Irrigation_Cost': float(irrigation_cost),
        'Pesticide_Usage': float(pesticide_usage),
        'Machinery_Cost': float(machinery_cost),
        'Transportation_Cost': float(transportation_cost)
    }])
    
    prediction = model.predict(input_data)[0]
    return round(float(prediction), 2)

if __name__ == "__main__":
    print("Testing dynamic prediction function on sample inputs...")
    sample_scenarios = [
        {
            'crop_type': 'Wheat',
            'farm_area': 5.0,
            'seed_cost': 9000.0,
            'fertilizer_usage': 600.0,
            'labor_requirements': 60.0,
            'irrigation_cost': 12500.0,
            'pesticide_usage': 12.5,
            'machinery_cost': 18500.0,
            'transportation_cost': 2800.0
        },
        {
            'crop_type': 'Sugarcane',
            'farm_area': 12.0,
            'seed_cost': 54000.0,
            'fertilizer_usage': 2640.0,
            'labor_requirements': 400.0,
            'irrigation_cost': 90000.0,
            'pesticide_usage': 54.0,
            'machinery_cost': 45000.0,
            'transportation_cost': 9200.0
        }
    ]
    
    for idx, s in enumerate(sample_scenarios, 1):
        cost = predict_production_cost(**s)
        print(f"\nScenario #{idx}: {s['crop_type']} on {s['farm_area']} acres")
        print(f" -> Predicted Total Production Cost: ₹{cost:,.2f}")
