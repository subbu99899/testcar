from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load trained model and columns
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

        # Convert JSON input into a DataFrame
        features = pd.DataFrame([data])

        # Convert categorical inputs to numerical values
        fuel_mapping = {"Petrol": 0, "Diesel": 1}
        transmission_mapping = {"Manual": 0, "Automatic": 1}

        if "fuel_type" in features:
            features["fuel_type"] = features["fuel_type"].map(fuel_mapping)
        if "Transmission" in features:
            features["Transmission"] = features["Transmission"].map(transmission_mapping)

        # One-hot encode the 'brand' column
        features = pd.get_dummies(features, columns=["brand"], drop_first=True)

        # Ensure columns match training data
        for col in model_columns:
            if col not in features:
                features[col] = 0  # Add missing columns with value 0

        # Reorder columns to match model's training data
        features = features[model_columns]

        # Make prediction
        prediction = model.predict(features)[0]

        return jsonify({"selling_price": round(prediction, 2)})
    
    except Exception as e:
        return jsonify({"error": str(e)})  # Return error message

if __name__ == "__main__":
    app.run(debug=True)
