import streamlit as st
import joblib
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Milk Yield Prediction",
    page_icon="🐄",
    layout="centered"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    
    /* Hide the default Streamlit header/footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Style input labels to be white and bold */
    .stNumberInput label {
        color: white !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    /* Style the input boxes */
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 12px;
        background: white;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 8px 20px rgba(245, 87, 108, 0.3);
    }
    .suggestion-box {
        background: white;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
    .heat-stress-low {
        background: linear-gradient(135deg, #48dbfb 0%, #0abde3 100%);
    }
    .heat-stress-moderate {
        background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%);
    }
    .heat-stress-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
    }
    
    /* Style the button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 15px 30px;
        font-size: 18px;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Remove padding from block container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
    }
    
    /* Style columns */
    [data-testid="column"] {
        background: transparent;
    }
</style>
""", unsafe_allow_html=True)

import zipfile
import os
import joblib
import streamlit as st

# Load model and scaler (with zip extraction)
@st.cache_resource
def load_models():
    # If models folder doesn't have the files, extract from zip
    if not os.path.exists("models/milk_yield_model.pkl") or not os.path.exists("models/scaler.pkl"):
        with zipfile.ZipFile("models.zip", "r") as zip_ref:
            zip_ref.extractall("models")
    
    # Load the model and scaler
    model = joblib.load("models/milk_yield_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

model, scaler = load_models()


def calculate_heat_stress(thi):
    """Calculate heat stress level based on THI"""
    if thi < 68:
        return "Low 🟢", "heat-stress-low"
    elif 68 <= thi < 72:
        return "Mild 🟡", "heat-stress-moderate"
    elif 72 <= thi < 80:
        return "Moderate 🟠", "heat-stress-moderate"
    else:
        return "Severe 🔴", "heat-stress-high"

def get_milk_yield_suggestions(prediction, lactation, dim, eating, ruminating):
    """Generate suggestions to improve milk yield"""
    suggestions = []
    
    if prediction < 20:
        suggestions.append("⚠️ Low milk yield detected.")
    
    if eating < 200:
        suggestions.append("• Increase feeding time - current eating time is below optimal (aim for 250+ minutes).")
    
    if ruminating < 400:
        suggestions.append("• Improve ruminating time - cows should ruminate 400-500 minutes daily for better digestion.")
    
    if lactation > 2:
        suggestions.append("• Late lactation stage detected - consider nutritional supplements to maintain production.")
    
    if dim > 200:
        suggestions.append("• High DIM value - monitor cow's health and consider dry period planning.")
    
    if not suggestions:
        suggestions.append("✅ All parameters are within optimal range! Continue current management practices.")
    
    return "<br>".join(suggestions)

def get_heat_stress_suggestions(stress_level, thi, rh):
    """Generate suggestions for managing heat stress"""
    suggestions = {
        "Low 🟢": "✅ Minimal heat stress. Maintain current cooling systems and monitor temperature changes.",
        "Mild 🟡": "⚠️ Mild heat stress detected. Ensure adequate water supply and good ventilation. Consider fans in barn areas.",
        "Moderate 🟠": "🔶 Moderate heat stress! Implement cooling measures:<br>• Install sprinklers<br>• Provide shade<br>• Increase water access<br>• Adjust feeding times to cooler periods",
        "Severe 🔴": "🚨 SEVERE HEAT STRESS! URGENT ACTION REQUIRED:<br>• Activate all cooling systems<br>• Provide constant water access<br>• Reduce stocking density<br>• Feed during night hours<br>• Monitor cows closely for heat stroke symptoms"
    }
    
    base_suggestion = suggestions.get(stress_level, "Monitor environmental conditions.")
    
    if rh > 80:
        base_suggestion += f"<br><br>⚠️ High humidity ({rh}%) reduces cooling efficiency - prioritize air movement."
    
    return base_suggestion

# Main app
def main():
    # Header
    st.markdown("""
        <div style='background: white; padding: 30px; border-radius: 20px; margin-bottom: 30px; box-shadow: 0 20px 60px rgba(0,0,0,0.3);'>
            <h1 style='text-align: center; color: #333; margin-bottom: 10px;'>🐄 Milk Yield & Heat Stress Prediction</h1>
            <p style='text-align: center; color: #666; font-size: 16px;'>AI-powered dairy management system</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Input fields in two columns (no white container)
    col1, col2 = st.columns(2)
    
    with col1:
        thi = st.number_input("🌡️ Temperature-Humidity Index (THI)", 
                             min_value=0.0, max_value=100.0, value=70.0, step=0.1,
                             help="Typical range: 50-90")
        ruminating = st.number_input("🐮 Ruminating Time (minutes)", 
                                    min_value=0.0, max_value=1000.0, value=450.0, step=1.0,
                                    help="Optimal: 400-500 minutes/day")
        lactation = st.number_input("🥛 Lactation Stage", 
                                   min_value=1, max_value=4, value=2, step=1,
                                   help="Stage 1-4")
    
    with col2:
        rh = st.number_input("💧 Relative Humidity (%)", 
                            min_value=0.0, max_value=100.0, value=60.0, step=0.1,
                            help="Typical range: 40-90%")
        eating = st.number_input("🌾 Eating Time (minutes)", 
                                min_value=0.0, max_value=1000.0, value=250.0, step=1.0,
                                help="Optimal: 250-300 minutes/day")
        dim = st.number_input("📅 Days in Milk (DIM)", 
                             min_value=0, max_value=500, value=100, step=1,
                             help="Days since last calving")
    
    # Predict button
    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("🔮 Predict Milk Yield")
    
    # Prediction and results
    if predict_button:
        try:
            # Prepare input
            inputs = [thi, rh, ruminating, eating, lactation, dim]
            inputs_scaled = scaler.transform([inputs])
            
            # Make prediction
            prediction = model.predict(inputs_scaled)[0]
            
            # Calculate heat stress
            stress_level, stress_class = calculate_heat_stress(thi)
            
            # Generate suggestions
            milk_suggestions = get_milk_yield_suggestions(prediction, lactation, dim, eating, ruminating)
            heat_suggestions = get_heat_stress_suggestions(stress_level, thi, rh)
            
            # Display results with custom styling
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Milk yield prediction
            st.markdown(f"""
                <div class='prediction-box'>
                    🥛 Predicted Milk Yield: {round(prediction, 2)} Liters/Day
                </div>
            """, unsafe_allow_html=True)
            
            # Heat stress level
            st.markdown(f"""
                <div class='prediction-box {stress_class}'>
                    🌡️ Heat Stress Level: {stress_level} (THI: {thi})
                </div>
            """, unsafe_allow_html=True)
            
            # Suggestions
            st.markdown(f"""
                <div class='suggestion-box'>
                    <h3 style='color: #667eea; margin-bottom: 15px;'>💡 Suggestions to Improve Milk Yield</h3>
                    <p style='color: #333; line-height: 1.8;'>{milk_suggestions}</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class='suggestion-box'>
                    <h3 style='color: #667eea; margin-bottom: 15px;'>🌡️ Heat Stress Management</h3>
                    <p style='color: #333; line-height: 1.8;'>{heat_suggestions}</p>
                </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Please check your input values and try again.")

if __name__ == "__main__":
    main()
