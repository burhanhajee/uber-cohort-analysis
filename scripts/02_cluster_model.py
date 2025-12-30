import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os

# --- CONFIGURATION ---
# Define input and output files
INPUT_PATH = os.path.join("data", "processed", "training_data.csv")
OUTPUT_PATH = os.path.join("data", "processed", "training_data_with_clusters.csv")
MODEL_DIR = "models"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
KMEANS_PATH = os.path.join(MODEL_DIR, "kmeans_model.joblib")

# Create the models directory
os.makedirs(MODEL_DIR, exist_ok=True)

def train_cluster_model():
    print("--- Starting Model 1: Driver Segmentation (Clustering) ---")

    # 1. Load Data
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Input file not found at {INPUT_PATH}")
        return
    
    df = pd.read_csv(INPUT_PATH)
    print(f"Data Loaded. Shape: {df.shape}")

    # 2. Feature Selection
    # DROPPING 'acceptance_rate' because it is 96% correlated with 'trip_utilization_rate'.
    # We features but exclude identifiers (ID) and targets (Churned).
    features_to_cluster = [
        'avg_earnings_per_hour_online',
        'trip_utilization_rate',
        'surge_reliance_score',
        'premium_trip_ratio',
        'quest_completion_rate',
        'cancellation_rate'
    ]


    X = df[features_to_cluster].copy()

    # 3. Preprocessing (Scaling)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Features successfully scaled.")

    # 4. Train K-Means
    # K=4 based on our EDA findings
    k = 4
    print(f"Training K-Means model with K={k}...")
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    # 5. Assign Labels
    df['cluster_label'] = kmeans.labels_


    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(kmeans, KMEANS_PATH)

    df.to_csv(OUTPUT_PATH, index=False)
    
    print("\nCluster Centers (Mean Values):")
    print(df.groupby('cluster_label')[features_to_cluster].mean())

if __name__ == "__main__":
    train_cluster_model()