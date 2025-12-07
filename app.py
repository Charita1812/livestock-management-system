from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("models/milk_yield_model.pkl")
scaler = joblib.load("models/scaler.pkl")

def calculate_heat_stress(thi):
    """Calculate heat stress level based on THI"""
    if thi < 68:
        return "Low", "heat-stress-low"
    elif 68 <= thi < 72:
        return "Mild", "heat-stress-moderate"
    elif 72 <= thi < 80:
        return "Moderate", "heat-stress-moderate"
    else:
        return "Severe", "heat-stress-high"

def get_milk_yield_suggestions(prediction, lactation, dim, eating, ruminating):
    """Generate suggestions to improve milk yield"""
    suggestions = []
    
    if prediction < 20:
        suggestions.append("⚠️ Low milk yield detected.")
    
    if eating < 200:
        suggestions.append("Increase feeding time - current eating time is below optimal (aim for 250+ minutes).")
    
    if ruminating < 400:
        suggestions.append("Improve ruminating time - cows should ruminate 400-500 minutes daily for better digestion.")
    
    if lactation > 2:
        suggestions.append("Late lactation stage detected - consider nutritional supplements to maintain production.")
    
    if dim > 200:
        suggestions.append("High DIM value - monitor cow's health and consider dry period planning.")
    
    if not suggestions:
        suggestions.append("✅ All parameters are within optimal range! Continue current management practices.")
    
    return " ".join(suggestions)

def get_heat_stress_suggestions(stress_level, thi, rh):
    """Generate suggestions for managing heat stress"""
    suggestions = {
        "Low": "✅ Minimal heat stress. Maintain current cooling systems and monitor temperature changes.",
        "Mild": "⚠️ Mild heat stress detected. Ensure adequate water supply and good ventilation. Consider fans in barn areas.",
        "Moderate": "🔶 Moderate heat stress! Implement cooling measures: Install sprinklers, provide shade, increase water access, and adjust feeding times to cooler periods.",
        "Severe": "🚨 SEVERE HEAT STRESS! URGENT ACTION REQUIRED: Activate all cooling systems, provide constant water access, reduce stocking density, feed during night hours, and monitor cows closely for heat stroke symptoms."
    }
    
    base_suggestion = suggestions.get(stress_level, "Monitor environmental conditions.")
    
    if rh > 80:
        base_suggestion += f" High humidity ({rh}%) reduces cooling efficiency - prioritize air movement."
    
    return base_suggestion

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Get input values from form (in order: THI, RH, Ruminating, Eating, Lactation, DIM)
            thi = float(request.form['THI'])
            rh = float(request.form['RH'])
            ruminating = float(request.form['Ruminating'])
            eating = float(request.form['Eating'])
            lactation = float(request.form['Lactation'])
            dim = float(request.form['DIM'])
            
            inputs = [thi, rh, ruminating, eating, lactation, dim]

            # Scale input data
            inputs_scaled = scaler.transform([inputs])

            # Make prediction
            prediction = model.predict(inputs_scaled)[0]
            
            # Calculate heat stress
            stress_level, stress_class = calculate_heat_stress(thi)
            
            # Generate suggestions
            milk_suggestions = get_milk_yield_suggestions(prediction, lactation, dim, eating, ruminating)
            heat_suggestions = get_heat_stress_suggestions(stress_level, thi, rh)
            
            return render_template("index.html", 
                                 prediction_text=f"🥛 Predicted Milk Yield: {round(prediction, 2)} Liters/Day",
                                 heat_stress_text=f"🌡️ Heat Stress Level: {stress_level} (THI: {thi})",
                                 heat_stress_class=stress_class,
                                 milk_yield_suggestion=milk_suggestions,
                                 heat_stress_suggestion=heat_suggestions)
        except Exception as e:
            return render_template("index.html", 
                                 error=f"Invalid input. Please enter valid numbers. Error: {str(e)}")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
