import joblib

# Load trained model
model = joblib.load("irrigation_model.pkl")
encoder = joblib.load("plant_encoder.pkl")

# Select plant
plant = "Tomato"

# Sensor values
soil_moisture = 32
temperature = 29
humidity = 64
rain = 0
water_level = 72

# Encode plant
plant_encoded = encoder.transform([plant])[0]

# Prepare input
import pandas as pd

data = pd.DataFrame([[
    plant_encoded,
    soil_moisture,
    temperature,
    humidity,
    rain,
    water_level
]], columns=[
    "Plant",
    "SoilMoisture",
    "Temperature",
    "Humidity",
    "Rain",
    "WaterLevel"
])

# Predict
prediction = model.predict(data)[0]

print("\n----- AI IRRIGATION PREDICTION -----")
print("Plant:", plant)
print("Soil Moisture:", soil_moisture)
print("Temperature:", temperature)
print("Humidity:", humidity)
print("Rain:", rain)
print("Water Level:", water_level)

if prediction == 1:
    print("Prediction: IRRIGATION REQUIRED")
else:
    print("Prediction: IRRIGATION NOT REQUIRED")