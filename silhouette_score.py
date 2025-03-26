import numpy as np
import pandas as pd
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import random

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Load and preprocess data
def load_data(data_path):
    df = pd.read_parquet(data_path)
    # Clean column names
    def rename_column(col):
        if col.startswith("first_"): return col[len("first_"):]
        elif col.startswith("last_"): return col[len("last_"):]
        return col.replace("proportion", "%")
    df = df.rename(columns=rename_column)
    # Map state numbers to names
    state_mapping = {0: 'penetrating', 1: 'oscillating', 2: 'bouncing'}
    df["state"] = df["state"].map(state_mapping)
    return df

# Compute silhouette scores and plot
def analyze_silhouette_scores(X, cluster_range):
    max_k = max(cluster_range)
    print(f"Computing spectral embedding with n_components={max_k}...")
    
    # Compute spectral embedding once
    embedding_model = SpectralEmbedding(
        n_components=max_k,
        affinity='nearest_neighbors',
        n_neighbors=100,
        random_state=42
    )
    embedding = embedding_model.fit_transform(X)
    print("Spectral embedding computed.")
    
    # Function to compute silhouette score for a given k
    def compute_silhouette(k):
        try:
            embed_k = embedding[:, :k]
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(embed_k)
            return silhouette_score(X, labels)
        except Exception as e:
            print(f"Error with {k} clusters: {e}")
            return 0
    
    print(f"Computing silhouette scores for clusters {min(cluster_range)} to {max(cluster_range)}...")
    
    # Parallel computation of silhouette scores
    silhouette_scores = Parallel(n_jobs=-1)(
        delayed(compute_silhouette)(k) for k in cluster_range
    )
    silhouette_scores = np.array(silhouette_scores)
    
    # Find optimal k
    valid_scores = silhouette_scores > 0
    if valid_scores.any():
        best_idx = np.argmax(silhouette_scores[valid_scores])
        best_k = cluster_range[np.where(valid_scores)[0][best_idx]]
        max_score = silhouette_scores[valid_scores][best_idx]
    else:
        best_k = cluster_range[0]
        max_score = 0
    
    # Create research-level figure
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, silhouette_scores, 'bo-', label='Silhouette Score', linewidth=2)
    plt.axvline(x=best_k, color='red', linestyle='--', label=f'Optimal k = {best_k}')
    plt.xlabel('Number of Clusters', fontsize=14)
    plt.ylabel('Silhouette Score', fontsize=14)
    plt.title('Silhouette Analysis for Spectral Clustering', fontsize=16, pad=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('silhouette_analysis.png', dpi=300)
    plt.close()
    
    return best_k, max_score, silhouette_scores

# Load data and prepare features
df = load_data("final_states_clean.parquet")
features = ['primary_flow', 'pressure', 'density', 'particle_feed', 'diameter']
X = df[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
cluster_range = list(range(80, 131))

# Run analysis
best_k, best_score, silhouette_scores = analyze_silhouette_scores(X_scaled, cluster_range)

# Display results
print("\n=== Silhouette Analysis Results ===")
print(f"Optimal number of clusters: {best_k}")
print(f"Best silhouette score: {best_score:.4f}")
print("Figure saved as 'silhouette_analysis.png'")