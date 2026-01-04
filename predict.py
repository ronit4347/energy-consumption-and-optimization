import pandas as pd
import joblib

# Load model
model_bundle = joblib.load("electricity_cost_model.pkl")
model = model_bundle["model"]

# Example input
input_data = {
    'Total_Energy_Consumption_kWh': 35.0'
    'Indoor_Temperature_C': 23.0,
    'Outdoor_Temperature_C': 31.0,
    'Appliance_Power_Consumption_kWh': 20.0,
    'HVAC_Usage_kWh': 12.0,
    'Solar_Power_Generated_kWh': 4.5,
    'Occupancy': 3
}

# Predict
input_df = pd.DataFrame([input_data])
prediction = model.predict(input_df)[0]
print(f"⚡ Predicted Electricity Cost: ₹{prediction:.2f} per kWh")
