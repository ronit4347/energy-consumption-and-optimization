import streamlit as st
import pandas as pd
import joblib

# Load model and scores
model_bundle = joblib.load("electricity_cost_model.pkl")
model = model_bundle["model"]
scores = model_bundle["scores"]

# Page settings
st.set_page_config(page_title="Electricity Cost Estimator", page_icon="⚡", layout="wide")

# --- Sidebar UI ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1684/1684375.png", width=100)
st.sidebar.title("🧾 Input Energy Parameters")

energy = st.sidebar.number_input("🔋 Total Energy Consumption (kWh)", min_value=0.0, value=35.0)
indoor_temp = st.sidebar.number_input("🌡️ Indoor Temperature (°C)", min_value=-10.0, value=23.0)
outdoor_temp = st.sidebar.number_input("🌤️ Outdoor Temperature (°C)", min_value=-10.0, value=30.0)
appliance_power = st.sidebar.number_input("🔌 Appliance Power Consumption (kWh)", min_value=0.0, value=20.0)
hvac_usage = st.sidebar.number_input("❄️ HVAC Usage (kWh)", min_value=0.0, value=10.0)
solar_power = st.sidebar.number_input("☀️ Solar Power Generated (kWh)", min_value=0.0, value=5.0)
occupancy = st.sidebar.number_input("👨‍👩‍👧 Occupancy", min_value=0, step=1, value=3)

# --- Main Panel ---
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>⚡ Electricity Cost Estimator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict your estimated electricity cost per unit based on usage and environment.</p>", unsafe_allow_html=True)
st.markdown("---")

# Button and prediction
if st.sidebar.button("🚀 Predict Cost"):
    input_df = pd.DataFrame([{
        'Total_Energy_Consumption_kWh': energy,
        'Indoor_Temperature_C': indoor_temp,
        'Outdoor_Temperature_C': outdoor_temp,
        'Appliance_Power_Consumption_kWh': appliance_power,
        'HVAC_Usage_kWh': hvac_usage,
        'Solar_Power_Generated_kWh': solar_power,
        'Occupancy': occupancy
    }])

    prediction = model.predict(input_df)[0]

    # Display result
    st.success(f"💡 **Estimated Electricity Cost:** ₹{prediction:.2f} per kWh")
    st.balloons()

    # Show input summary
    with st.expander("🔍 See Input Summary"):
        st.table(input_df.T.rename(columns={0: "Value"}))

    # Show model scores
    with st.expander("📊 Model Accuracy (R² Score)"):
        for name, score in scores.items():
            st.write(f"**{name}**: {score:.4f}")
else:
    st.info("👈 Fill in the parameters on the left and click **Predict Cost** to see the result.")
