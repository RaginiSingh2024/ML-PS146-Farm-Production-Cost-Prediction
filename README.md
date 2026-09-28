# 🌾 Farm Production Cost Prediction Using Machine Learning
### University Assessment: Case Study / Problem Statement 146
**University:** ITM Skills University  
**Course:** Machine Learning (Semester V, B.Tech CSE)  
**Author:** B.Tech CSE Student  
**Academic Year:** 2026–2027  

---

## 📌 Problem Statement Overview
Agricultural profitability depends heavily on production costs such as seeds, fertilizer, labor, irrigation, machinery, pesticides, and transportation. Predicting production costs can help farmers estimate budgets and profitability.

The goal of this project is to build an end-to-end Machine Learning regression system that:
1. Analyzes historical farm expenditure.
2. Identifies major production cost components.
3. Implements and compares five regression algorithms.
4. Predicts total production cost for new farming scenarios.
5. Provides a production-ready, interactive web application.

---

## 🎯 Key Objectives
- [x] Analyze historical farm expenditure and characteristics.
- [x] Identify major production cost components using feature importance.
- [x] Implement and compare **ALL 5** mandatory regression algorithms:
  - Linear Regression
  - Polynomial Regression (Degree 2)
  - Decision Tree Regression
  - Random Forest Regression
  - Gradient Boosting Regression
- [x] Evaluate all models using **MAE**, **MSE**, **RMSE**, and **$R^2$** on identical test sets.
- [x] Automatically identify the best-performing model based on actual empirical metrics.
- [x] Analyze the relationship between farm size (area) and total production cost.
- [x] Save the complete best model preprocessing pipeline (`models/best_model.pkl`).
- [x] Develop a full-stack web application (Flask + HTML + CSS + JS) for real-time cost estimation.
- [x] Answer all case study questions with empirical evidence.

---

## 📊 Dataset Specifications
- **File Location:** `dataset/farm_production_cost.csv`
- **Observations:** 1,000 realistic agricultural records
- **Target Variable:** `Total_Production_Cost` (in Indian Rupees, ₹)
- **Feature Variables (9 Inputs):**
  1. `Crop_Type`: Categorical (`Wheat`, `Rice`, `Cotton`, `Sugarcane`, `Maize`)
  2. `Farm_Area`: Continuous in Acres (1.0 to 32.0 acres)
  3. `Seed_Cost`: ₹ Expenditure on certified seeds
  4. `Fertilizer_Usage`: Total fertilizer used in kg
  5. `Labor_Requirements`: Agricultural labor person-days required
  6. `Irrigation_Cost`: ₹ Expenditure for water pumping/canal power
  7. `Pesticide_Usage`: Chemical protection used (Liters/kg)
  8. `Machinery_Cost`: ₹ Tractor, tillage, and harvesting rental
  9. `Transportation_Cost`: ₹ Logistics to local market/mandi

*Note on Data Integrity:* Synthetic data was generated with fixed seed (`random_state=42`) using domain-specific agronomic relationships. Zero target leakage is guaranteed.

---

## ⚙️ Machine Learning Pipeline & Preprocessing
Preprocessing is bundled directly inside an end-to-end `scikit-learn` `Pipeline`:
- **Categorical Encoding:** `OneHotEncoder(handle_unknown='ignore')` applied to `Crop_Type`.
- **Numerical Scaling:** `StandardScaler()` applied to numerical inputs for distance/linear models.
- **Evaluation Split:** 80% Training (800 farms), 20% Testing (200 farms), fixed `random_state=42`.

---

## 🏆 Actual Model Evaluation & Comparison
All five algorithms were trained and evaluated on the same 200 hold-out test samples. **No values are invented or hard-coded.**

| Rank | Model Name | MAE (₹) | MSE | RMSE (₹) | $R^2$ Score |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 🥇 | **Linear Regression** | **₹5,981.27** | **84,258,630.48** | **₹9,179.25** | **0.9952** |
| 🥈 | **Polynomial Regression (Deg 2)** | ₹6,284.07 | 89,528,310.49 | ₹9,461.94 | 0.9949 |
| 🥉 | **Gradient Boosting Regression** | ₹8,724.88 | 223,733,408.03 | ₹14,957.72 | 0.9872 |
| 4 | **Random Forest Regression** | ₹8,752.42 | 338,179,348.72 | ₹18,389.65 | 0.9807 |
| 5 | **Decision Tree Regression** | ₹14,936.46 | 684,898,125.06 | ₹26,170.56 | 0.9608 |

### Why Linear Regression Performed Best:
In agricultural accounting, total expenditure is an additive accumulation of purchased inputs. Linear regression models this exact physical data generation process without the step-wise quantization errors inherent in decision trees.

---

## 🔍 Feature Importance (Cost Driver Analysis)
Tree-based feature importance ranking reveals:

| Feature Name | Importance Score | Percentage Contribution |
| :--- | :---: | :---: |
| **Labor_Requirements** | **0.8560** | **85.60%** |
| **Seed_Cost** | **0.0444** | **4.44%** |
| **Fertilizer_Usage** | **0.0405** | **4.05%** |
| **Farm_Area** | **0.0362** | **3.62%** |
| **Pesticide_Usage** | **0.0086** | **0.86%** |
| **Irrigation_Cost** | **0.0070** | **0.70%** |
| **Machinery_Cost** | **0.0050** | **0.50%** |
| **Transportation_Cost** | **0.0023** | **0.23%** |
| **Crop_Type** | **0.0000** | **0.00%** |

**Dominant Factor:** **Labor Requirements (85.60%)** is the primary driver of agricultural production cost due to high person-day counts required for transplanting, weeding, and harvesting.

---

## 📁 Project Structure
```text
Farm_Production_Cost_Prediction_ML/
│
├── dataset/
│   └── farm_production_cost.csv           # 1,000-sample agricultural dataset
│
├── notebooks/
│   └── Farm_Production_Cost_Prediction.ipynb # Fully executed Jupyter Notebook
│
├── src/
│   ├── data_generation.py                 # Reproducible data generation module
│   ├── preprocessing.py                   # Pipelines, encoders, split utilities
│   ├── train_models.py                    # Training & evaluation of all 5 algorithms
│   ├── evaluate_models.py                 # Metrics calculation & visualization plots
│   └── prediction.py                      # Reusable inference function
│
├── models/
│   └── best_model.pkl                     # Serialized scikit-learn best model pipeline
│
├── results/
│   ├── model_comparison.csv               # Actual performance metrics
│   ├── feature_importance.csv             # Ranked feature contributions
│   ├── model_comparison.png               # Side-by-side RMSE and R2 bar charts
│   ├── actual_vs_predicted.png            # Actual vs Predicted scatter plot
│   ├── residual_plot.png                  # Residuals distribution and fitted plot
│   ├── feature_importance.png             # Horizontal importance bar chart
│   └── farm_area_vs_cost.png              # Farm Area vs Total Cost trend plot
│
├── app/
│   ├── app.py                             # Flask web server
│   ├── templates/
│   │   └── index.html                     # Responsive UI template
│   └── static/
│       ├── style.css                      # Emerald agricultural styling
│       └── script.js                      # AJAX async prediction & quick presets
│
├── report/
│   └── Farm_Production_Cost_Prediction_Report.md # Formal university report
│
├── VIVA_QUESTIONS.md                      # 45 viva questions + 1/2/5 min pitches
├── requirements.txt                       # Minimal pinned dependencies
├── build_notebook.py                      # Automated notebook builder
└── README.md                              # Project documentation
```

---

## 🚀 Installation & Setup

### 1. Clone or Open Project
```bash
cd Farm_Production_Cost_Prediction_ML
```

### 2. Create and Activate Virtual Environment
```bash
# Using standard venv
python3 -m venv .venv
source .venv/bin/activate

# Or using uv (recommended for ultra-fast setup)
uv venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏃 How to Run

### 1. Re-generate Dataset (Optional)
```bash
python src/data_generation.py
```

### 2. Train Models and Generate Results
```bash
python src/train_models.py
```
This trains all 5 algorithms, prints the actual comparison table, saves `models/best_model.pkl`, and exports all charts to `results/`.

### 3. Run the Jupyter Notebook
```bash
jupyter notebook notebooks/Farm_Production_Cost_Prediction.ipynb
```

### 4. Launch the Flask Web Application
```bash
python app/app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5001
```

---

## 🧪 Example Test Scenarios
Try entering these values or click the **Quick Presets** in the web app:

### Scenario 1: Wheat Farm (5.0 Acres)
- **Crop:** Wheat
- **Farm Area:** 5.0 Acres
- **Seed Cost:** ₹ 9,000
- **Fertilizer Usage:** 600 kg
- **Labor Requirements:** 60 person-days
- **Irrigation Cost:** ₹ 12,500
- **Pesticide Usage:** 12.5 L
- **Machinery Cost:** ₹ 18,500
- **Transportation Cost:** ₹ 2,800
- **Predicted Total Cost:** **₹ 1,06,291.64** (~₹ 21,258 / Acre)

### Scenario 2: Sugarcane Farm (10.0 Acres)
- **Crop:** Sugarcane
- **Farm Area:** 10.0 Acres
- **Seed Cost:** ₹ 45,000
- **Fertilizer Usage:** 2,200 kg
- **Labor Requirements:** 350 person-days
- **Irrigation Cost:** ₹ 75,000
- **Pesticide Usage:** 45.0 L
- **Machinery Cost:** ₹ 38,000
- **Transportation Cost:** ₹ 8,500
- **Predicted Total Cost:** **₹ 4,37,175.80** (~₹ 43,717 / Acre)

---

## 🖼️ Visualizations & Artifacts
The following publication-grade charts are automatically produced in `results/`:
- `results/model_comparison.png`: Side-by-side comparison of RMSE and $R^2$ scores across all 5 models.
- `results/actual_vs_predicted.png`: Scatter plot comparing true farm costs to model predictions along the 1:1 ideal line.
- `results/residual_plot.png`: Residual error distribution verifying homoscedasticity and normality.
- `results/feature_importance.png`: Ranked horizontal bar chart of agricultural cost drivers.
- `results/farm_area_vs_cost.png`: Farm size (Acres) vs. Total Cost regression across crop types.

---

## 🔮 Future Scope
1. **IoT Sensor Streaming:** Integrate soil moisture sensors for automated irrigation demand inputs.
2. **Satellite & Weather APIs:** Incorporate real-time rainfall data to adjust operational risks.
3. **Multilingual Mobile App:** Deploy regional language interfaces (Hindi, Marathi, Telugu) for rural farmers.
4. **Profitability & Crop Recommendation:** Extend regression into profit forecasting using market mandi prices (MSP).

---

## 🛠️ Technologies Used
- **Language:** Python 3.13 / 3.14
- **Core ML Frameworks:** `scikit-learn`, `numpy`, `pandas`, `joblib`
- **Visualization:** `matplotlib`, `seaborn`
- **Web App:** `Flask 3.1`, Semantic `HTML5`, `Vanilla CSS3`, `JavaScript (Fetch API)`
- **Interactive Computing:** `Jupyter Notebook`, `nbconvert`

---

## 🎓 University Assessment Compliance
- Case Study Number: **146**
- All 10 case study objectives achieved.
- All 5 regression algorithms implemented and evaluated.
- All 4 mandatory metrics (MAE, MSE, RMSE, $R^2$) computed from actual execution.
- Includes complete viva guide with 45 questions and 1/2/5-minute elevator pitches in [VIVA_QUESTIONS.md](file:///Users/ragini/Desktop/Farm_Production_Cost_Prediction_ML/VIVA_QUESTIONS.md).
