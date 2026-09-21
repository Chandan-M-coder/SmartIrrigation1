import serial
import pandas as pd
import joblib

# Load ML model
model = joblib.load("irrigation_model.pkl")
encoder = joblib.load("plant_encoder.pkl")

# Connect to Heltec Gateway
ser = serial.Serial("COM3", 115200, timeout=1)

print("===================================")
print(" AI SMART IRRIGATION SYSTEM")
print("===================================")
print("Waiting for field node data...")
print()

while True:

    line = ser.readline().decode("utf-8", errors="ignore").strip()

    if not line:
        continue

    print("Received:", line)

    # Ignore other gateway messages
    if "," not in line:
        continue

    try:

        # Expected format:
        # Tomato,32,29,64,0,72

        plant, soil, temperature, humidity, rain, water = line.split(",")

        soil = float(soil)
        temperature = float(temperature)
        humidity = float(humidity)
        rain = int(rain)
        water = float(water)

        # Convert plant name to encoded number
        plant_encoded = encoder.transform([plant])[0]

        # Create DataFrame
        data = pd.DataFrame([[
            plant_encoded,
            soil,
            temperature,
            humidity,
            rain,
            water
        ]], columns=[
            "Plant",
            "SoilMoisture",
            "Temperature",
            "Humidity",
            "Rain",
            "WaterLevel"
        ])

        # ML prediction
        prediction = model.predict(data)[0]

        print("-----------------------------------")
        print("Plant          :", plant)
        print("Soil Moisture  :", soil, "%")
        print("Temperature    :", temperature, "°C")
        print("Humidity       :", humidity, "%")
        print("Rain           :", rain)
        print("Water Level    :", water, "%")

        if prediction == 1:
            print("AI DECISION    : IRRIGATION REQUIRED")
        else:
            print("AI DECISION    : IRRIGATION NOT REQUIRED")

        print("-----------------------------------")
        print()

    except Exception as e:
        print("Data processing error:", e)