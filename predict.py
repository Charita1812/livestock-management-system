import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

# Load trained model and scaler
model = joblib.load("models/milk_yield_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Define the required features (must match training)
feature_columns = ["THI", "RH (%)", "Ruminating", "Eating", "Lactation", "DIM"]

app = Flask(__name__)

def predict_milk_yield(features):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([features], columns=feature_columns)
        
        # Scale input features
        features_scaled = scaler.transform(df)

        # Predict milk yield
        milk_yield_prediction = model.predict(features_scaled)[0]
        milk_yield_prediction = round(float(milk_yield_prediction), 2)  # Ensure it's a valid float

        # Heat Stress Classification
        THI = features[0]  # First value in input
        if THI < 72:
            heat_stress = "Low"
            heat_stress_suggestion = "No immediate actions needed. Keep monitoring for any changes."
        elif 72 <= THI <= 78:
            heat_stress = "Moderate"
            heat_stress_suggestion = "Consider providing extra shade and ensuring access to cool water."
        else:
            heat_stress = "High"
            heat_stress_suggestion = "Implement cooling measures such as fans or sprinklers and reduce feed intake during peak heat."

        # Milk Yield Suggestions
        if milk_yield_prediction < 10:
            milk_yield_suggestion = "Consider improving feed quality (e.g., high-protein feed) and check for lactation-related health issues."
            lactation_suggestion = "For cows in early lactation, ensure a high-quality, energy-dense diet to support milk production."
            health_suggestion = "Check for signs of metabolic issues (e.g., ketosis) or mastitis that could reduce milk production."
        elif milk_yield_prediction < 20:
            milk_yield_suggestion = "Monitor diet to ensure balanced nutrition. Ensure proper hydration and assess stress factors."
            lactation_suggestion = "In mid-lactation, provide a balanced diet with a mix of roughage and concentrates to maintain steady milk production."
            health_suggestion = "Keep an eye on the cow's health for any signs of infection or injury that might affect milk yield."
        else:
            milk_yield_suggestion = "Good milk yield. Maintain cow health and nutrition for continued high production."
            lactation_suggestion = "For late lactation cows, ensure they have access to sufficient feed and care."
            health_suggestion = "Continue regular health check-ups to sustain milk yield."

        return (
            milk_yield_prediction,
            heat_stress,
            heat_stress_suggestion,
            milk_yield_suggestion,
            lactation_suggestion,
            health_suggestion,
        )

    except Exception as e:
        return None, "Error", str(e), "Invalid input. Please enter numeric values.", "", ""

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Get user input
            features = [
                float(request.form["THI"]),
                float(request.form["RH"]),
                float(request.form["Ruminating"]),
                float(request.form["Eating"]),
                float(request.form["Lactation"]),
                float(request.form["DIM"]),
            ]

            # Get predictions and suggestions
            (
                milk_yield_prediction,
                heat_stress,
                heat_stress_suggestion,
                milk_yield_suggestion,
                lactation_suggestion,
                health_suggestion,
            ) = predict_milk_yield(features)

            return render_template(
                "index.html",
                prediction_text=f"Predicted Milk Yield: {milk_yield_prediction} liters",
                heat_stress_text=f"Heat Stress Level: {heat_stress}",
                heat_stress_suggestion=heat_stress_suggestion,
                milk_yield_suggestion=milk_yield_suggestion,
                lactation_suggestion=lactation_suggestion,
                health_suggestion=health_suggestion,
            )

        except ValueError:
            return render_template("index.html", prediction_text="Invalid input. Please enter numeric values.")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
