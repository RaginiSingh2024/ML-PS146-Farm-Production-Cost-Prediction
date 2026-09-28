# ITM SKILLS UNIVERSITY
## School of Computer Science & Engineering
### Academic Assessment: Machine Learning Laboratory (Semester V)
---
# Academic Project Report
## Farm Production Cost Prediction Using Machine Learning
**Problem Statement / Case Study Number:** 146  
**Course:** Machine Learning (Semester V, B.Tech CSE)  
**Academic Year:** 2026–2027  
**Submission Date:** September 28, 2026  

---

## Abstract
Agricultural profitability and financial stability depend heavily on managing farm operational expenditures, including seed procurement, fertilizer application, irrigation pumping, mechanization, human labor, chemical pesticides, and mandi transportation. Accurate pre-season production cost forecasting enables smallholder farmers, corporate agribusinesses, and rural lending institutions to allocate capital efficiently, prevent distress financing, and formulate realistic crop budgets. 

This project develops an end-to-end Machine Learning regression framework for agricultural production cost estimation using historical and simulated farm expenditure data. Five regression algorithms were rigorously trained, evaluated, and cross-compared under identical 80-20 train-test conditions: **Linear Regression**, **Polynomial Regression (Degree 2)**, **Decision Tree Regression**, **Random Forest Regression**, and **Gradient Boosting Regression**. Model performance was assessed using Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and the Coefficient of Determination ($R^2$). 

Empirical results demonstrate that **Linear Regression** achieved the highest accuracy, with an $R^2$ of **0.9952** and a minimal Root Mean Squared Error of **₹9,179.25** on the 200 hold-out test instances. Tree-based feature importance analysis revealed that **Labor Requirements** represents the single most dominant cost driver (contributing **85.60%** of total variance), followed by seed costs and fertilizer inputs. The trained pipeline was integrated into a production-ready web application using Python Flask, vanilla CSS, and JavaScript, providing real-time inference and agricultural budgeting recommendations for practical deployment.

---

## 1. Introduction
Agriculture represents the foundation of the Indian rural economy, providing livelihoods to over 50% of the working population. However, farm income volatility has increased markedly due to rising input prices, unpredictable climatic conditions, and shifting market dynamics. A recurring challenge faced by farming households is the absence of systematic, data-driven cost estimation mechanisms prior to sowing.

Traditional agricultural budgeting relies on ad-hoc estimations, verbal quotes from local input dealers, or historical memory. These rudimentary approaches frequently fail to account for non-linear scale effects, labor wage inflation, and crop-specific input intensities. Machine learning provides an empirical methodology to model these complex relationships, learn from historical expenditure patterns, and output calibrated financial predictions.

This report documents the design, empirical development, statistical evaluation, and web deployment of a machine learning regression system to predict total farm production costs.

---

## 2. Problem Statement
**Case Study Number: 146**  
Agricultural profitability depends on production costs such as seeds, fertilizer, labor, irrigation, machinery, pesticides, and transportation. Predicting production costs can help farmers estimate budgets and profitability. The goal is to build a Machine Learning based regression system that analyzes historical farm expenditure, identifies major cost components, compares multiple regression algorithms, predicts total production cost for a new farming scenario, and provides a usable web application.

---

## 3. Project Objectives
1. **Analyze Historical Farm Expenditure:** Examine the statistical properties, variances, and correlations of farm cost components.
2. **Identify Major Cost Components:** Use feature importance and regression analysis to rank expenditure drivers.
3. **Develop Production-Cost Prediction Models:** Formulate end-to-end machine learning pipelines.
4. **Compare Five Regression Algorithms:** Implement and cross-evaluate Linear Regression, Polynomial Regression, Decision Tree Regression, Random Forest, and Gradient Boosting.
5. **Estimate Expected Production Cost for New Farming Scenarios:** Provide a reusable, dynamic inference interface.
6. **Identify the Optimal Model:** Objectively select the highest-performing model based on actual test error metrics.
7. **Analyze Feature Influence:** Determine the mathematical contribution of each farming variable.
8. **Analyze Farm Size Relationships:** Evaluate how total expenditure and per-acre cost behave across varying farm sizes.
9. **Support Agricultural Budgeting:** Translate predictive outputs into actionable budgeting guidance.
10. **Deploy Interactive Web Application:** Implement a modern, responsive web application for real-time cost estimation.

---

## 4. Dataset Description
Because the university problem statement does not provide a pre-packaged public dataset, a realistic, reproducible agricultural dataset of 1,000 farm records was generated with a fixed seed (`random_state=42`) using domain-specific agronomic relationships.

### Variables & Specifications:
1. **Crop_Type** (Categorical): The primary crop cultivated (`Wheat`, `Rice`, `Cotton`, `Sugarcane`, `Maize`).
2. **Farm_Area** (Continuous, Float): Total cultivated land in acres (range: 1.0 to 32.0 acres).
3. **Seed_Cost** (Continuous, Float): Expenditure incurred on certified seed stock (₹).
4. **Fertilizer_Usage** (Continuous, Float): Total chemical/organic fertilizer applied (kg).
5. **Labor_Requirements** (Continuous, Float): Total human labor person-days required for sowing, weeding, spraying, and harvesting.
6. **Irrigation_Cost** (Continuous, Float): Expenditure for canal water access, tube-well electricity, and diesel pump fuel (₹).
7. **Pesticide_Usage** (Continuous, Float): Total quantity of liquid/powder crop protection formulations applied (Liters/kg).
8. **Machinery_Cost** (Continuous, Float): Rental and operating costs for tractors, rotavators, and harvesters (₹).
9. **Transportation_Cost** (Continuous, Float): Logistics expenditure for moving harvest to local mandis and wholesale procurement centers (₹).
10. **Total_Production_Cost** (Continuous, Float — **Target Variable**): Total financial expenditure incurred across the crop season (₹).

### Data Integrity Rules Enforced:
- **Reproducibility:** Seed fixed at 42.
- **Physical Relationships:** Seed costs, irrigation, and labor rates reflect documented per-acre crop profiles (e.g., Sugarcane and Rice require significantly higher water and fertilizer than Wheat and Maize).
- **Economies of Scale:** Mechanization and labor requirements incorporate slight non-linear efficiency gains for larger land holdings.
- **No Target Leakage:** Total_Production_Cost is not a direct arithmetic sum of feature columns; unit usages (kg, person-days, liters) are converted via market wage and price models with operational overhead and realistic stochastic environmental variance.

---

## 5. Data Preprocessing
Data preprocessing was implemented using `scikit-learn`'s `ColumnTransformer` to guarantee zero data leakage between training and testing sets.

1. **Missing Values & Duplicate Audit:**
   - Missing values check: 0 missing cells across all 10 columns.
   - Duplicate records check: 0 duplicate rows.
2. **Categorical Encoding:**
   - `Crop_Type` was encoded using `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
3. **Feature Scaling:**
   - Numerical input features (`Farm_Area`, `Seed_Cost`, `Fertilizer_Usage`, etc.) were standardized using `StandardScaler` ($\mu = 0, \sigma = 1$) for parametric models (Linear Regression, Polynomial Regression) to ensure uniform gradient descent and prevent coefficient distortion.
4. **Train-Test Partitioning:**
   - An 80:20 train-test split was executed using `train_test_split(random_state=42)`.
   - Training sample size: 800 farms.
   - Testing sample size: 200 farms.

---

## 6. Exploratory Data Analysis (EDA)
Comprehensive exploratory analysis was conducted to examine distribution morphology and feature interactions:

1. **Target Distribution:** `Total_Production_Cost` exhibits a right-skewed log-normal-type distribution, reflecting the reality that the majority of farms are small-to-medium land holdings (1–10 acres), while a smaller subset comprises large commercial holdings (10–30+ acres). Mean cost was ₹1,87,861 with a median of ₹1,49,635.
2. **Farm Area Correlation:** A high Pearson correlation coefficient ($r \approx 0.98$) was observed between `Farm_Area` and `Total_Production_Cost`, confirming that acreage represents the primary scale dimension in agricultural spending.
3. **Crop-Wise Cost Profiles:**
   - **Sugarcane:** Highest average expenditure per acre (~₹42,000/acre) due to an extended 12-month cultivation cycle, intense irrigation requirements (₹7,500/acre), and heavy manual harvesting labor (35 person-days/acre).
   - **Cotton:** High labor expenditure (28 person-days/acre for picking) and high pesticide costs (7.5 L/acre).
   - **Wheat & Maize:** Moderate total cost per acre (~₹21,000/acre) facilitated by higher tractor mechanization and shorter crop cycles.

---

## 7. Machine Learning Algorithms Implemented

### 7.1 Linear Regression
Linear Regression models the relationship between target $y$ and feature vector $X$ via a linear combination of weighted parameters:
$$y = \beta_0 + \sum_{i=1}^p \beta_i X_i + \epsilon$$
It minimizes Ordinary Least Squares (OLS) residual sum of squares:
$$J(\beta) = \sum_{i=1}^n (y_i - \hat{y}_i)^2$$
Because agricultural costs represent the linear accumulation of purchased inputs and hired resources, Linear Regression provides a mathematically sound, highly interpretable baseline.

### 7.2 Polynomial Regression (Degree 2)
Polynomial Regression extends linear regression by introducing interaction terms and quadratic terms:
$$y = \beta_0 + \sum \beta_i X_i + \sum \beta_{ij} X_i X_j + \sum \beta_{ii} X_i^2$$
Using degree $d=2$ on 13 preprocessed features generated 104 interaction features. Regularization via $L_2$ Ridge penalty ($\alpha = 1.0$) was incorporated to prevent collinearity and numerical instability.

### 7.3 Decision Tree Regression
Decision Tree Regression partitions the feature space into orthogonal multidimensional rectangular regions using recursive binary splitting. The criterion minimizes mean squared error:
$$MSE = \frac{1}{N_m} \sum_{i \in R_m} (y_i - \hat{y}_{R_m})^2$$
Hyperparameters were bounded (`max_depth=6`, `min_samples_split=10`, `min_samples_leaf=5`) to prevent overfitting.

### 7.4 Random Forest Regression
Random Forest is an ensemble bagging meta-estimator constructing an ensemble of 120 de-correlated decision trees:
- **Bootstrap Aggregation:** Each tree is trained on a distinct bootstrap sample of the training records.
- **Random Subspace Method:** At each node split, a random subset of features is considered.
- **Aggregation:** Predictions are averaged across all trees: $\hat{y} = \frac{1}{B}\sum_{b=1}^B T_b(x)$.
This reduces model variance while maintaining low bias.

### 7.5 Gradient Boosting Regression
Gradient Boosting builds an additive ensemble sequentially:
$$F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)$$
Each new weak learner $h_m(x)$ is fitted to the pseudo-residuals of the existing ensemble:
$$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$$
Configured with `n_estimators=150`, `learning_rate=0.08`, and `max_depth=4`.

---

## 8. Evaluation Metrics & Model Comparison

### 8.1 Metric Definitions
1. **Mean Absolute Error (MAE):** Average magnitude of absolute prediction errors in currency units (₹).
   $$MAE = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
2. **Mean Squared Error (MSE):** Average of squared prediction discrepancies, penalizing large outliers heavily.
   $$MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$
3. **Root Mean Squared Error (RMSE):** Square root of MSE, measuring error in direct currency units (₹).
   $$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
4. **Coefficient of Determination ($R^2$):** Proportion of variance in target explained by model predictors.
   $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

### 8.2 Actual Evaluation Results (Hold-Out Test Set: 200 Records)
The following metrics were computed from actual model execution on unseen test data:

| Rank | Model Name | MAE (₹) | MSE | RMSE (₹) | $R^2$ Score |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **Linear Regression** | **₹5,981.27** | **84,258,630.48** | **₹9,179.25** | **0.9952** |
| 2 | Polynomial Regression (Degree 2) | ₹6,284.07 | 89,528,310.49 | ₹9,461.94 | 0.9949 |
| 3 | Gradient Boosting Regression | ₹8,724.88 | 223,733,408.03 | ₹14,957.72 | 0.9872 |
| 4 | Random Forest Regression | ₹8,752.42 | 338,179,348.72 | ₹18,389.65 | 0.9807 |
| 5 | Decision Tree Regression | ₹14,936.46 | 684,898,125.06 | ₹26,170.56 | 0.9608 |

---

## 9. Feature Importance Analysis
Using the Gradient Boosting and Random Forest feature importance evaluators (MDI / Gini impurity reduction), the relative contribution of each agricultural input was quantified:

| Feature Name | Normalized Importance | Relative Share (%) |
| :--- | :---: | :---: |
| **Labor_Requirements** | **0.8560** | **85.60%** |
| **Seed_Cost** | **0.0444** | **4.44%** |
| **Fertilizer_Usage** | **0.0405** | **4.05%** |
| **Farm_Area** | **0.0362** | **3.62%** |
| **Pesticide_Usage** | **0.0086** | **0.86%** |
| **Irrigation_Cost** | **0.0070** | **0.70%** |
| **Machinery_Cost** | **0.0050** | **0.50%** |
| **Transportation_Cost**| **0.0023** | **0.23%** |
| **Crop_Type** | **0.0000** | **0.00%** |

### Insight:
**Labor Requirements** has by far the greatest influence on agricultural production cost (**85.60%**). Agricultural operations in India remain labor-intensive, requiring manual weeding, transplanting, pruning, and harvesting. Even modest fluctuations in daily labor wages or person-days disproportionately sway total cost.

---

## 10. Farm Area vs. Cost Analysis
Analysis confirms an approximately linear scaling relationship between `Farm_Area` and `Total_Production_Cost` ($R^2 > 0.96$). However, the per-acre cost curve exhibits slight economies of scale:
- Farms smaller than 3 acres incur a higher per-acre machinery mobilization overhead (~₹24,500/acre).
- Farms larger than 15 acres achieve lower unit operational costs (~₹20,800/acre) due to bulk fertilizer purchases and consolidated equipment utilization.

---

## 11. Best Model Selection & Justification
**Selected Model:** **Linear Regression** (encapsulated in an end-to-end `Pipeline` with `StandardScaler` and `OneHotEncoder`).

### Why Linear Regression Outperformed Ensembles:
1. **Mathematical Nature of Production Accounting:** Production costs represent an additive aggregation of unit expenditures (Seed + Labor + Irrigation + Fertilizer + Mechanization). Linear models naturally match this underlying physical data-generating process.
2. **Absence of Arbitrary Step Discontinuities:** Tree models partition continuous inputs into step-wise piecewise constant intervals, introducing quantization error on smooth additive processes.
3. **Superior Generalization:** Linear Regression achieved the lowest test RMSE (**₹9,179.25**) without exhibiting overfitting or tree variance.

The complete pipeline was serialized to disk at `models/best_model.pkl`.

---

## 12. Web Application Architecture
A full-stack, presentation-ready web application was created using:
- **Backend:** Python Flask 3.1.x
- **Frontend Structure:** Semantic HTML5 (`app/templates/index.html`)
- **Styling:** Custom Vanilla CSS3 with emerald agricultural design system (`app/static/style.css`)
- **Interactivity:** JavaScript Fetch API for real-time AJAX inference (`app/static/script.js`)

### Web Features:
- Dropdowns and numeric fields for all 9 required inputs.
- Quick Preset buttons for Wheat, Rice, Cotton, Sugarcane, and Maize for one-click demonstration.
- Real-time client and server validation (checks positive area, non-negative costs, valid crop choices).
- Dynamic result cards displaying Total Production Cost (₹), Cost per Acre (₹), and active model metrics.
- Embedded model performance comparison table and feature importance indicators.

---

## 13. Answers to Core University Assessment Questions

### Question 1: Can farm production costs be predicted?
**Yes.** Machine learning models achieved an $R^2$ of **0.9952** and a Mean Absolute Error of **₹5,981.27** on hold-out testing data, proving that farm costs can be predicted with high precision.

### Question 2: Which cost component has the greatest influence?
**Labor Requirements** has the greatest influence, accounting for **85.60%** of total model importance. Human labor person-days represent the largest financial volatility point in crop management.

### Question 3: Which regression model performs best?
**Linear Regression** performed best, achieving the lowest RMSE (**₹9,179.25**), lowest MAE (**₹5,981.27**), and highest $R^2$ (**0.9952**).

### Question 4: How does farm size affect total cost?
Total production cost scales proportionally with farm acreage, but per-acre unit costs decrease slightly on larger farms due to economies of scale in mechanization and input discounts.

### Question 5: Can production costs be estimated for a new farm?
**Yes.** The saved pipeline processes any arbitrary combination of crop type, acreage, and operational inputs, returning an instant, validated cost projection.

### Question 6: Can ML-based cost prediction support farm budget planning?
**Yes.** Pre-season cost forecasts allow farmers to secure accurate crop loans, negotiate bulk input contracts, allocate seasonal working capital, and avoid distress borrowing.

---

## 14. Limitations
1. **Simulated Agronomic Baseline:** Although realistic, the dataset is synthetic and does not capture hyper-local geo-climatic anomalies or real-time daily mandi wage shifts.
2. **Macroeconomic Shocks:** Extreme events such as sudden fertilizer export bans, diesel fuel subsidies removal, or pest epidemics may cause unexpected cost surges outside the model's bounds.
3. **Regional Wage Variation:** The model assumes uniform state-level daily wage averages rather than village-level micro-rates.

---

## 15. Future Scope
1. **IoT Sensor Integration:** Connecting automated soil moisture sensors and drone imagery to predict fertilizer and irrigation requirements automatically.
2. **Satellite & Weather Forecasting APIs:** Integrating real-time precipitation forecasts to adjust irrigation cost estimates dynamically.
3. **Multi-Regional Calibration:** Expanding the dataset to incorporate distinct agricultural agro-climatic zones across different Indian states.
4. **Mobile Application Development:** Deploying an offline-capable React Native / Flutter Android application for rural farmers in regional languages (Hindi, Marathi, Telugu).

---

## 16. Conclusion
This project successfully achieved all 10 objectives mandated by Case Study 146 of ITM Skills University. Five machine learning regression algorithms were implemented and empirically compared. Linear Regression emerged as the optimal model ($R^2 = 0.9952$, RMSE = ₹9,179.25), driven primarily by Labor Requirements (85.60%). A full-stack web application was deployed, providing an end-to-end practical solution for agricultural budgeting.

---

## 17. References
1. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825-2830.
2. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.
3. Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine*. Annals of Statistics, 1189-1232.
4. Directorate of Economics and Statistics, Ministry of Agriculture & Farmers Welfare, Government of India. *Cost of Cultivation of Principal Crops*.
5. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*. Springer.
