import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
import random

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Load data function
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

# Load your actual data
df = load_data("final_states_clean.parquet")

# Define features to use directly from your dataframe
features = ['primary_flow', 'pressure', 'density', 'particle_feed', 'diameter']

# Check if all features exist in the dataframe
for feature in features:
    if feature not in df.columns:
        print(f"Warning: Column '{feature}' not found in the dataframe")

print(f"Using features: {features}")
print(f"Available columns: {df.columns.tolist()}")

# Extract features and labels
X = df[features].values
y = df['state'].map({'penetrating': 0, 'oscillating': 1, 'bouncing': 2}).values  # Map states to integers

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define hyperparameter grid
n_clusters_list = [8, 20 , 120] # Number of clusters to test
n_neighbors_list = [100]  # For affinity='nearest_neighbors'
affinities = ['nearest_neighbors']  # Only using nearest_neighbors affinity

# Function to calculate purity for each cluster
def calculate_purity(cluster_labels, true_labels):
    unique_clusters = np.unique(cluster_labels)
    purity_results = []
    for cluster in unique_clusters:
        cluster_indices = np.where(cluster_labels == cluster)[0]
        cluster_states = true_labels[cluster_indices]
        if len(cluster_states) == 0:
            continue
        state_counts = np.bincount(cluster_states, minlength=3)  # Assuming 3 states (0, 1, 2)
        dominant_state = np.argmax(state_counts)
        purity = state_counts[dominant_state] / len(cluster_states) * 100
        purity_results.append((cluster, dominant_state, purity, len(cluster_states)))
    return purity_results

# Test different configurations
results = []  # Store all results for later analysis
print(f"Total samples: {len(X)}")
print(f"Distribution of states: {pd.Series(df['state']).value_counts().to_dict()}")

for affinity in affinities:
    print(f"\n### Testing Affinity: {affinity} ###")
    param_name = 'n_neighbors'
    param_values = n_neighbors_list
    
    for n_clusters in n_clusters_list:
        for param in param_values:
            # Configure Spectral Clustering
            clustering = SpectralClustering(
                n_clusters=n_clusters,
                affinity=affinity,
                n_neighbors=param,
                random_state=42,
                assign_labels='kmeans'  # Using kmeans for label assignment
            )
            
            # Perform clustering
            try:
                labels = clustering.fit_predict(X_scaled)
                
                # Calculate purity
                purity_results = calculate_purity(labels, y)
                
                # Display results for this configuration
                print(f"\nTesting n_clusters={n_clusters}, {param_name}={param}")
                
                # First, calculate overall purity stats
                cluster_sizes = {state: 0 for state in ['penetrating', 'oscillating', 'bouncing']}
                high_purity_clusters = {state: 0 for state in ['penetrating', 'oscillating', 'bouncing']}
                found_high_purity = False
                
                for cluster, dominant_state, purity, size in purity_results:
                    state_name = {0: 'penetrating', 1: 'oscillating', 2: 'bouncing'}[dominant_state]
                    results.append({
                        'affinity': affinity,
                        'n_clusters': n_clusters,
                        param_name: param,
                        'cluster': cluster,
                        'dominant_state': state_name,
                        'purity': purity,
                        'size': size
                    })
                    
                    cluster_sizes[state_name] += size
                    if purity >= 90:
                        high_purity_clusters[state_name] += 1
                    
                    if purity >= 95 and state_name in ['penetrating', 'bouncing']:
                        print(f"Cluster {cluster}: {purity:.2f}% {state_name} (size: {size})")
                        found_high_purity = True
                
                # Print summary
                print("Summary:")
                total_samples = sum(cluster_sizes.values())
                for state, size in cluster_sizes.items():
                    percentage = (size / total_samples) * 100 if total_samples > 0 else 0
                    print(f"  {state}: {size} samples ({percentage:.1f}%), {high_purity_clusters[state]} high-purity clusters")
                
                if not found_high_purity:
                    print("No clusters with ≥ 95% purity for 'penetrating' or 'bouncing'.")
            
            except Exception as e:
                print(f"Error with configuration n_clusters={n_clusters}, {param_name}={param}: {str(e)}")

# Save results to CSV for further analysis
results_df = pd.DataFrame(results)
results_df.to_csv('spectral_clustering_results.csv', index=False)
print("\nResults saved to 'spectral_clustering_results.csv'")

# Show best clusters for each state
print("\n=== Best Clusters by State ===")
for state in ['penetrating', 'oscillating', 'bouncing']:
    state_results = results_df[results_df['dominant_state'] == state]
    if not state_results.empty:
        best_result = state_results.sort_values(['purity', 'size'], ascending=[False, False]).iloc[0]
        print(f"\nBest cluster for '{state}':")
        print(f"  Affinity: {best_result['affinity']}")
        print(f"  n_clusters: {best_result['n_clusters']}")
        if 'n_neighbors' in best_result:
            print(f"  n_neighbors: {best_result['n_neighbors']}")
        print(f"  Cluster: {best_result['cluster']}")
        print(f"  Purity: {best_result['purity']:.2f}%")
        print(f"  Size: {best_result['size']} samples")

# Show most balanced configuration
print("\n=== Most Balanced Configuration ===")
# Group by configuration parameters
config_groups = results_df.groupby(['n_clusters', 'n_neighbors'])
# Find configurations with at least one high-purity cluster for each state
balanced_configs = []

for (n_clusters, n_neighbors), group in config_groups:
    state_purities = {}
    for state in ['penetrating', 'oscillating', 'bouncing']:
        state_group = group[group['dominant_state'] == state]
        if not state_group.empty:
            best_purity = state_group['purity'].max()
            best_size = state_group.loc[state_group['purity'].idxmax(), 'size']
            state_purities[state] = (best_purity, best_size)
    
    # Check if we have results for all three states
    if len(state_purities) == 3:
        # Calculate a balance score - higher is better
        min_purity = min(p[0] for p in state_purities.values())
        avg_purity = sum(p[0] for p in state_purities.values()) / 3
        balance_score = min_purity * 0.7 + avg_purity * 0.3  # Weight min purity more
        
        balanced_configs.append({
            'n_clusters': n_clusters,
            'n_neighbors': n_neighbors,
            'min_purity': min_purity,
            'avg_purity': avg_purity,
            'balance_score': balance_score,
            'penetrating': state_purities.get('penetrating', (0, 0)),
            'oscillating': state_purities.get('oscillating', (0, 0)),
            'bouncing': state_purities.get('bouncing', (0, 0))
        })

if balanced_configs:
    # Sort by balance score
    balanced_configs.sort(key=lambda x: x['balance_score'], reverse=True)
    best_config = balanced_configs[0]
    
    print(f"Most balanced configuration:")
    print(f"  n_clusters: {best_config['n_clusters']}")
    print(f"  n_neighbors: {best_config['n_neighbors']}")
    print(f"  Min cluster purity: {best_config['min_purity']:.2f}%")
    print(f"  Avg cluster purity: {best_config['avg_purity']:.2f}%")
    print(f"  State purities:")
    for state in ['penetrating', 'oscillating', 'bouncing']:
        purity, size = best_config[state]
        print(f"    {state}: {purity:.2f}% (size: {size})")
else:
    print("Could not find a balanced configuration with all three states represented.")
