import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import joblib

# Load dataset
df = pd.read_csv("smart_home_energy_consumption.csv")

# Define input features and target
features = [
    'Total_Energy_Consumption_kWh',
    'Indoor_Temperature_C',
    'Outdoor_Temperature_C',
    'Appliance_Power_Consumption_kWh',
    'HVAC_Usage_kWh',
    'Solar_Power_Generated_kWh',
    'Occupancy'
]
target = 'Electricity_Cost_INR_per_kWh'

# Prepare training data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define individual models
xgb = XGBRegressor()
rf = RandomForestRegressor()
lr = LinearRegression()

# Fit individual models
xgb.fit(X_train, y_train)
rf.fit(X_train, y_train)
lr.fit(X_train, y_train)

# Calculate R^2 scores
xgb_score = r2_score(y_test, xgb.predict(X_test))
rf_score = r2_score(y_test, rf.predict(X_test))
lr_score = r2_score(y_test, lr.predict(X_test))

print("🎯 Model Scores:")
print(f"XGBoost R²: {xgb_score:.4f}")
print(f"Random Forest R²: {rf_score:.4f}")
print(f"Linear Regression R²: {lr_score:.4f}")

# Create ensemble model
ensemble_model = VotingRegressor(estimators=[
    ('xgb', xgb),
    ('rf', rf),
    ('lr', lr)
])
ensemble_model.fit(X_train, y_train)

# Save model and scores
joblib.dump({
    "model": ensemble_model,
    "scores": {
        "XGBoost": xgb_score,
        "RandomForest": rf_score,
        "LinearRegression": lr_score
    }
}, "electricity_cost_model.pkl")

print("✅ Ensemble model and scores saved successfully.")
