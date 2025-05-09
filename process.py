import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess_data(csv_file):
    # Load dataset
    df = pd.read_csv(csv_file)

    # Select only the required features
    selected_features = ["THI", "RH (%)", "Ruminating", "Eating", "Lactation", "DIM", "MilkYield"]
    
    # Ensure column names match
    df = df[selected_features]

    # Handle missing values (if any)
    df = df.dropna()

    # Separate features and target variable
    X = df.drop(columns=["MilkYield"])
    y = df["MilkYield"]

    # Save preprocessed data for training
    X.to_csv("processed_features.csv", index=False)
    y.to_csv("processed_target.csv", index=False)

    print("Data processing complete. Saved processed data.")

if __name__ == "__main__":
    preprocess_data("data/totmerge.csv")  # Change this to your actual CSV file
