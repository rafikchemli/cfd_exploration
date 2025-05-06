#!/usr/bin/env python3
"""
Create clustered data with tracker ID

This script loads particle data, performs spectral clustering, and saves the results
to a CSV file including tracker IDs.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
import random

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

def load_data(data_path):
    df = pd.read_parquet(data_path)
    def rename_column(col):
        if col.startswith("first_"): return col[len("first_"):]
        elif col.startswith("last_"): return col[len("last_"):]
        return col.replace("proportion", "%")
    df = df.rename(columns=rename_column)
    state_mapping = {0: 'penetrating', 1: 'oscillating', 2: 'bouncing'}
    df["state"] = df["state"].map(state_mapping)
    return df

def perform_clustering(X_scaled, n_clusters):
    print(f"Performing Spectral clustering with n_clusters={n_clusters}, random_state=42")
    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity='nearest_neighbors',
        n_neighbors=100,
        random_state=42,
        assign_labels='kmeans'
    )
    return clustering.fit_predict(X_scaled)

def calculate_proportions(df):
    """Calculate the proportion of each state for each source file."""
    # Group by source file and calculate state proportions
    proportions = df.groupby('source_file')['state'].value_counts(normalize=True).unstack(fill_value=0)
    proportions = proportions.reset_index()
    
    # Rename columns to match the expected format
    proportion_cols = {state: f'proportion_{state}' for state in ['penetrating', 'oscillating', 'bouncing']}
    proportions = proportions.rename(columns=proportion_cols)
    
    # Merge proportions back to the main dataframe
    df = df.merge(proportions, on='source_file')
    
    return df

def main():
    # Load and prepare data
    print("Loading data from final_states_clean.parquet")
    df = load_data("final_states_clean.parquet")
    
    # Define features to use based on Clustering_figures.py
    features = ['diameter', 'density', 'pressure', 'primary_flow', 'particle_feed']
    
    # Verify features exist in the dataframe
    for feature in features:
        if feature not in df.columns:
            print(f"Warning: Column '{feature}' not found in the dataframe")
    
    print(f"Using features: {features}")
    
    # Extract features
    X = df[features].values
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Set number of clusters to 100 based on the request
    n_clusters = 100
    
    # Perform clustering
    df['cluster'] = perform_clustering(X_scaled, n_clusters)
    
    # Calculate state proportions per source file and add to dataframe
    df = calculate_proportions(df)
    
    # Save the result to CSV
    output_file = "final_states_wcluster_tracker.csv"
    print(f"Saving results to {output_file}")
    df.to_csv(output_file, index=False)
    
    # Print cluster distribution
    print("Cluster distribution:")
    print(df['cluster'].value_counts().sort_index().to_dict())
    
    print(f"Completed! Data saved to {output_file}")

if __name__ == "__main__":
    main()