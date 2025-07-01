from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load trained model and model columns
model = joblib.load("car_price_model.pkl")
model_columns = joblib.load("model_columns.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Convert input to DataFrame
        features = pd.DataFrame([data])

        # Map categorical inputs to numerical values
        fuel_mapping = {"Petrol": 0, "Diesel": 1}
        transmission_mapping = {"Manual": 0, "Automatic": 1}

        if "fuel_type" in features:
            features["fuel_type"] = features["fuel_type"].map(fuel_mapping)
        if "Transmission" in features:
            features["Transmission"] = features["Transmission"].map(transmission_mapping)

        # One-hot encode the 'brand' column
        features = pd.get_dummies(features, columns=["brand"], drop_first=True)

        # Add missing columns with 0
        for col in model_columns:
            if col not in features:
                features[col] = 0

        # Reorder columns to match model input
        features = features[model_columns]

        # Make prediction
        prediction = model.predict(features)[0]

        return jsonify({"selling_price": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": str(e)})

# ✅ Render Deployment Fix — Use correct host and port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render sets the PORT env variable
    app.run(host="0.0.0.0", port=port, debug=True)
