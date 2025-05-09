from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("models/milk_yield_model.pkl")
scaler = joblib.load("models/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Get input values from form
            inputs = [float(request.form[key]) for key in request.form]

            # Scale input data
            inputs_scaled = scaler.transform([inputs])

            # Make prediction
            prediction = model.predict(inputs_scaled)[0]
            return render_template("index.html", result=round(prediction, 2))
        except:
            return render_template("index.html", error="Invalid input. Please enter valid numbers.")

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
