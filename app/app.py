"""
app/app.py
Flask Web Application for Farm Production Cost Prediction.
Case Study 146 - Machine Learning Semester V - ITM Skills University.
"""

import os
import sys

# Ensure project root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from flask import Flask, render_template, request, jsonify
import pandas as pd
from src.prediction import load_best_model, predict_production_cost

app = Flask(__name__)

# Valid crops
VALID_CROPS = ['Wheat', 'Rice', 'Cotton', 'Sugarcane', 'Maize']

# Sample presets for quick testing
SAMPLE_PRESETS = {
    'wheat': {
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
    'rice': {
        'crop_type': 'Rice',
        'farm_area': 6.0,
        'seed_cost': 13200.0,
        'fertilizer_usage': 900.0,
        'labor_requirements': 132.0,
        'irrigation_cost': 39000.0,
        'pesticide_usage': 24.0,
        'machinery_cost': 22500.0,
        'transportation_cost': 3800.0
    },
    'cotton': {
        'crop_type': 'Cotton',
        'farm_area': 7.5,
        'seed_cost': 27000.0,
        'fertilizer_usage': 975.0,
        'labor_requirements': 210.0,
        'irrigation_cost': 24000.0,
        'pesticide_usage': 56.0,
        'machinery_cost': 27000.0,
        'transportation_cost': 3600.0
    },
    'sugarcane': {
        'crop_type': 'Sugarcane',
        'farm_area': 10.0,
        'seed_cost': 45000.0,
        'fertilizer_usage': 2200.0,
        'labor_requirements': 350.0,
        'irrigation_cost': 75000.0,
        'pesticide_usage': 45.0,
        'machinery_cost': 38000.0,
        'transportation_cost': 8500.0
    },
    'maize': {
        'crop_type': 'Maize',
        'farm_area': 4.0,
        'seed_cost': 8000.0,
        'fertilizer_usage': 440.0,
        'labor_requirements': 56.0,
        'irrigation_cost': 8800.0,
        'pesticide_usage': 12.0,
        'machinery_cost': 15000.0,
        'transportation_cost': 2400.0
    }
}

# Preload best model at server start
model = load_best_model()

# Load model comparison and feature importance if available
def get_model_summary():
    comp_path = os.path.join(BASE_DIR, "results", "model_comparison.csv")
    imp_path = os.path.join(BASE_DIR, "results", "feature_importance.csv")
    
    comp_data = []
    if os.path.exists(comp_path):
        comp_df = pd.read_csv(comp_path)
        comp_data = comp_df.to_dict(orient='records')
        
    imp_data = []
    if os.path.exists(imp_path):
        imp_df = pd.read_csv(imp_path)
        imp_data = imp_df.to_dict(orient='records')
        
    return comp_data, imp_data

@app.route('/', methods=['GET'])
def index():
    comp_data, imp_data = get_model_summary()
    return render_template(
        'index.html',
        crops=VALID_CROPS,
        model_comparison=comp_data,
        feature_importance=imp_data
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Support both JSON payload and standard form-encoded data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        # Extract and validate Crop_Type
        crop_type = str(data.get('crop_type', '')).strip().capitalize()
        if not crop_type or crop_type not in VALID_CROPS:
            return jsonify({
                'success': False,
                'error': f"Invalid crop type '{crop_type}'. Please choose from: {', '.join(VALID_CROPS)}."
            }), 400

        # Validate numeric inputs
        numeric_fields = {
            'farm_area': ('Farm Area', 0.1, 500.0),
            'seed_cost': ('Seed Cost', 0.0, 1000000.0),
            'fertilizer_usage': ('Fertilizer Usage', 0.0, 50000.0),
            'labor_requirements': ('Labor Requirements', 0.0, 10000.0),
            'irrigation_cost': ('Irrigation Cost', 0.0, 1000000.0),
            'pesticide_usage': ('Pesticide Usage', 0.0, 5000.0),
            'machinery_cost': ('Machinery Cost', 0.0, 1000000.0),
            'transportation_cost': ('Transportation Cost', 0.0, 500000.0)
        }

        parsed_values = {}
        for key, (label, min_val, max_val) in numeric_fields.items():
            raw_val = data.get(key)
            if raw_val is None or str(raw_val).strip() == '':
                return jsonify({
                    'success': False,
                    'error': f"Field '{label}' is required."
                }), 400
            
            try:
                val = float(raw_val)
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': f"'{label}' must be a valid numeric value."
                }), 400

            if val < min_val:
                return jsonify({
                    'success': False,
                    'error': f"'{label}' cannot be less than {min_val}."
                }), 400
                
            if val > max_val:
                return jsonify({
                    'success': False,
                    'error': f"'{label}' exceeds realistic threshold ({max_val})."
                }), 400

            parsed_values[key] = val

        # Perform ML prediction using the trained best model
        predicted_cost = predict_production_cost(
            crop_type=crop_type,
            farm_area=parsed_values['farm_area'],
            seed_cost=parsed_values['seed_cost'],
            fertilizer_usage=parsed_values['fertilizer_usage'],
            labor_requirements=parsed_values['labor_requirements'],
            irrigation_cost=parsed_values['irrigation_cost'],
            pesticide_usage=parsed_values['pesticide_usage'],
            machinery_cost=parsed_values['machinery_cost'],
            transportation_cost=parsed_values['transportation_cost'],
            model=model
        )

        cost_per_acre = round(predicted_cost / parsed_values['farm_area'], 2)

        return jsonify({
            'success': True,
            'crop_type': crop_type,
            'farm_area': parsed_values['farm_area'],
            'predicted_cost': predicted_cost,
            'formatted_cost': f"₹ {predicted_cost:,.2f}",
            'cost_per_acre': cost_per_acre,
            'formatted_cost_per_acre': f"₹ {cost_per_acre:,.2f}"
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f"An error occurred during prediction: {str(e)}"
        }), 500

@app.route('/api/preset/<crop_name>', methods=['GET'])
def get_preset(crop_name):
    crop_key = crop_name.lower().strip()
    if crop_key in SAMPLE_PRESETS:
        return jsonify({
            'success': True,
            'preset': SAMPLE_PRESETS[crop_key]
        })
    return jsonify({
        'success': False,
        'error': f"Preset for '{crop_name}' not found."
    }), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"Starting Farm Production Cost Prediction Web App on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
