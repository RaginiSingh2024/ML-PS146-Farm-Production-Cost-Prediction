"""
src/data_generation.py
Module to generate a realistic, reproducible agricultural production cost dataset
for university ML assessment (Case Study 146).
"""

import os
import numpy as np
import pandas as pd

def generate_farm_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate realistic agricultural production expenditure data.
    
    Parameters:
    -----------
    n_samples : int, default=1000
        Number of farm records to simulate.
    random_state : int, default=42
        Seed for reproducibility.
        
    Returns:
    --------
    pd.DataFrame containing 9 feature columns and 1 target column:
        - Crop_Type: Categorical (Wheat, Rice, Cotton, Sugarcane, Maize)
        - Farm_Area: Continuous in Acres (1.0 to 30.0 acres)
        - Seed_Cost: ₹ Total cost of certified seeds
        - Fertilizer_Usage: Total fertilizer used (in kg)
        - Labor_Requirements: Labor person-days utilized
        - Irrigation_Cost: ₹ Total irrigation/fuel/pump cost
        - Pesticide_Usage: Total chemical crop protection used (in Liters/kg)
        - Machinery_Cost: ₹ Tractor, tilling, and harvest mechanization cost
        - Transportation_Cost: ₹ Logistics to market/mandi
        - Total_Production_Cost: ₹ Total farm expenditure (Target variable)
    """
    np.random.seed(random_state)
    
    crops = ['Wheat', 'Rice', 'Cotton', 'Sugarcane', 'Maize']
    crop_probs = [0.25, 0.25, 0.20, 0.15, 0.15]
    
    selected_crops = np.random.choice(crops, size=n_samples, p=crop_probs)
    
    # Farm area: Log-normal distribution reflecting realistic land holding patterns (1 to ~30 acres)
    raw_area = np.random.lognormal(mean=1.6, sigma=0.65, size=n_samples)
    farm_area = np.clip(raw_area, 1.0, 32.0).round(2)
    
    # Crop-specific characteristic parameters (rates per acre)
    # Seed cost per acre (₹)
    seed_rate_mean = {
        'Wheat': 1800, 'Rice': 2200, 'Cotton': 3600, 'Sugarcane': 4500, 'Maize': 2000
    }
    # Fertilizer usage per acre (kg)
    fert_rate_mean = {
        'Wheat': 120, 'Rice': 150, 'Cotton': 130, 'Sugarcane': 220, 'Maize': 110
    }
    # Labor person-days per acre
    labor_rate_mean = {
        'Wheat': 12, 'Rice': 22, 'Cotton': 28, 'Sugarcane': 35, 'Maize': 14
    }
    # Irrigation cost per acre (₹)
    irrig_rate_mean = {
        'Wheat': 2500, 'Rice': 6500, 'Cotton': 3200, 'Sugarcane': 7500, 'Maize': 2200
    }
    # Pesticide usage per acre (Liters)
    pest_rate_mean = {
        'Wheat': 2.5, 'Rice': 4.0, 'Cotton': 7.5, 'Sugarcane': 4.5, 'Maize': 3.0
    }
    
    seed_costs = []
    fertilizer_usages = []
    labor_requirements = []
    irrigation_costs = []
    pesticide_usages = []
    machinery_costs = []
    transportation_costs = []
    total_production_costs = []
    
    for i in range(n_samples):
        c = selected_crops[i]
        area = farm_area[i]
        
        # 1. Seed cost with realistic variation
        seed_cost_per_acre = np.random.normal(seed_rate_mean[c], seed_rate_mean[c] * 0.08)
        seed_cost = max(500.0, seed_cost_per_acre * area)
        
        # 2. Fertilizer usage (kg)
        fert_per_acre = np.random.normal(fert_rate_mean[c], fert_rate_mean[c] * 0.10)
        fert_usage = max(20.0, fert_per_acre * area)
        
        # 3. Labor requirements (person-days)
        labor_per_acre = np.random.normal(labor_rate_mean[c], labor_rate_mean[c] * 0.12)
        # Slight labor efficiency on larger farms
        labor_req = max(5.0, (labor_per_acre * area) * (1.0 - 0.005 * min(area, 20)))
        
        # 4. Irrigation cost (₹)
        irrig_per_acre = np.random.normal(irrig_rate_mean[c], irrig_rate_mean[c] * 0.12)
        irrig_cost = max(800.0, irrig_per_acre * area)
        
        # 5. Pesticide usage (Liters/kg)
        pest_per_acre = np.random.normal(pest_rate_mean[c], pest_rate_mean[c] * 0.15)
        pest_usage = max(1.0, pest_per_acre * area)
        
        # 6. Machinery cost (₹)
        # Fixed initial mobilization cost + per acre rate with economies of scale
        base_machinery_per_acre = 3500.0 + np.random.normal(0, 300)
        # Economies of scale: cost per acre decreases slightly with area
        machinery_cost = max(2000.0, (base_machinery_per_acre * area * (1.0 - 0.008 * min(area, 25))) + 1500.0)
        
        # 7. Transportation cost (₹)
        # Tied to area, crop harvest bulkiness, and distance factor
        crop_bulk_factor = {'Wheat': 1.0, 'Rice': 1.1, 'Cotton': 0.8, 'Sugarcane': 1.6, 'Maize': 1.0}
        distance_km = np.random.uniform(8.0, 45.0)
        transp_cost = max(600.0, (area * 320.0 * crop_bulk_factor[c]) + (distance_km * 45.0) + np.random.normal(200, 50))
        
        # 8. Target: Total_Production_Cost (₹)
        # In economics:
        # Total cost = Seed Cost + (Fertilizer Usage * fertilizer price per kg)
        #              + (Labor Requirements * daily agricultural wage rate)
        #              + Irrigation Cost + (Pesticide Usage * pesticide price per liter)
        #              + Machinery Cost + Transportation Cost
        #              + Land preparation / miscellaneous farm overhead + realistic market/weather variance
        
        avg_fertilizer_price_per_kg = 28.5 + np.random.normal(0, 1.2)   # e.g., ₹28.5/kg blended
        avg_daily_wage = 450.0 + np.random.normal(0, 15.0)              # e.g., ₹450/day agricultural labor
        avg_pesticide_price_per_l = 850.0 + np.random.normal(0, 35.0)   # e.g., ₹850/L formulation
        
        direct_inputs = (
            seed_cost + 
            (fert_usage * avg_fertilizer_price_per_kg) + 
            (labor_req * avg_daily_wage) + 
            irrig_cost + 
            (pest_usage * avg_pesticide_price_per_l) + 
            machinery_cost + 
            transp_cost
        )
        
        # Operational overhead (storage, insurance, equipment maintenance, soil health tests)
        operational_overhead = (1200.0 * area) + 2500.0
        
        # Realistic environmental and market fluctuation (unobserved factors ~ 3-5%)
        noise = np.random.normal(0, direct_inputs * 0.035)
        
        total_cost = direct_inputs + operational_overhead + noise
        
        seed_costs.append(round(seed_cost, 2))
        fertilizer_usages.append(round(fert_usage, 2))
        labor_requirements.append(round(labor_req, 1))
        irrigation_costs.append(round(irrig_cost, 2))
        pesticide_usages.append(round(pest_usage, 2))
        machinery_costs.append(round(machinery_cost, 2))
        transportation_costs.append(round(transp_cost, 2))
        total_production_costs.append(round(total_cost, 2))
        
    df = pd.DataFrame({
        'Crop_Type': selected_crops,
        'Farm_Area': farm_area,
        'Seed_Cost': seed_costs,
        'Fertilizer_Usage': fertilizer_usages,
        'Labor_Requirements': labor_requirements,
        'Irrigation_Cost': irrigation_costs,
        'Pesticide_Usage': pesticide_usages,
        'Machinery_Cost': machinery_costs,
        'Transportation_Cost': transportation_costs,
        'Total_Production_Cost': total_production_costs
    })
    
    return df

def save_dataset(output_path: str = "dataset/farm_production_cost.csv", n_samples: int = 1000):
    """Generate and save the agricultural dataset."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = generate_farm_data(n_samples=n_samples, random_state=42)
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Dataset successfully generated with {len(df)} records and saved to '{output_path}'.")
    print(f"Columns: {list(df.columns)}")
    print(f"Summary statistics of Total_Production_Cost:\n{df['Total_Production_Cost'].describe()}")
    return df

if __name__ == "__main__":
    save_dataset()
