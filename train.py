import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Load processed data
X = pd.read_csv("processed_features.csv")
y = pd.read_csv("processed_target.csv")

# Ensure target column is numeric
y = y.squeeze()

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(model, "models/milk_yield_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Training complete. Model and scaler saved.")
