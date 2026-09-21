from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)
# Load existing trained model
model = joblib.load("irrigation_model.pkl")

# Load existing plant encoder
encoder = joblib.load("plant_encoder.pkl")


@app.route("/")
def home():
    return jsonify({
        "status": "success",
        "message": "Smart Irrigation ML API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        # Get values from website
        plant = data["plant"]
        soil_moisture = float(data["soil_moisture"])
        temperature = float(data["temperature"])
        humidity = float(data["humidity"])
        rain = float(data["rain"])
        water_level = float(data["water_level"])

        # Convert plant name to encoded value
        plant_encoded = encoder.transform([plant])[0]

        # Same columns used during training
        input_data = pd.DataFrame([[
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

        # Existing Random Forest model
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            result = "IRRIGATION REQUIRED"
        else:
            result = "IRRIGATION NOT REQUIRED"

        return jsonify({
            "plant": plant,
            "soil_moisture": soil_moisture,
            "temperature": temperature,
            "humidity": humidity,
            "rain": rain,
            "water_level": water_level,
            "prediction": result
        })

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


if __name__ == "__main__":

    import os

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )