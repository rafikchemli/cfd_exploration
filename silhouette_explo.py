import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# Define the features to analyze
features = ['diameter', 'density', 'pressure', 'primary_flow', 'particle_feed']

# Load your data
df = pd.read_csv('final_states_with_clusters.csv')

# Scale features with MinMaxScaler (0 to 1)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df[features])
for i, feature in enumerate(features):
    df[f'scaled_{feature}'] = X_scaled[:, i]

# Streamlit app
st.title("Cluster Feature Analysis")

# Create two columns for Cluster Selection and Sort Clusters
col1, col2 = st.columns(2)

# Cluster Selection in the first column
with col1:
    st.header("Cluster Selection")
    selected_state = st.selectbox("Select state", options=['penetrating', 'oscillating', 'bouncing'])

# Sort Clusters in the second column
with col2:
    st.header("Sort Clusters")
    sort_options = ["State Proportion"] + features
    sort_by = st.selectbox("Sort clusters by", options=sort_options, index=0)  # Default to "State Proportion"

num_clusters = st.slider("Number of top clusters", 1, len(df['cluster'].unique()), 3)
# Find top clusters by state proportion
cluster_props = df.groupby('cluster')['state'].value_counts(normalize=True).unstack(fill_value=0) * 100
top_clusters = cluster_props.sort_values(selected_state, ascending=False).head(num_clusters).index.tolist()
st.write(f"Top clusters for {selected_state}: {', '.join(map(str, top_clusters))}")

# Filter data for top clusters
filtered_df = df[df['cluster'].isin(top_clusters)]

# Calculate mean scaled features (features as rows, clusters as columns)
cluster_stats = filtered_df.groupby('cluster')[[f'scaled_{feature}' for feature in features]].mean()
cluster_stats = cluster_stats.reindex(top_clusters).T  # Transpose: features on y-axis
cluster_stats.index = [f'scaled_{feature}' for feature in features]  # Full feature names as index

# Calculate state proportions
props = filtered_df.groupby('cluster')['state'].value_counts(normalize=True).unstack(fill_value=0) * 100

# Sort the clusters based on the selected option
if sort_by == "State Proportion":
    sorted_clusters = props.sort_values(selected_state, ascending=False).index
else:
    sort_values = cluster_stats.loc[f'scaled_{sort_by}']
    sorted_clusters = sort_values.sort_values(ascending=False).index

# Reorder cluster_stats and props based on sorted clusters
cluster_stats_sorted = cluster_stats[sorted_clusters]
props_sorted = props.loc[sorted_clusters]

# Create cluster labels for display
cluster_labels = [f"Cluster {c}" for c in sorted_clusters]

# Display heatmap
st.subheader(f"Mean Scaled Features Sorted by {sort_by}")
fig = go.Figure(data=go.Heatmap(
    z=cluster_stats_sorted.values,
    x=cluster_labels,
    y=features,  # Use original feature names for readability
    text=np.round(cluster_stats_sorted.values, 2),
    texttemplate='%{text}',
    colorscale='Viridis',
    zmin=0, zmax=1  # Matches MinMaxScaler range
))
fig.update_layout(
    title=f'Top {num_clusters} {selected_state} Clusters Sorted by {sort_by}',
    xaxis_title='Cluster',
    yaxis_title='Scaled Feature',
    height=500,
    width=800
)
st.plotly_chart(fig)

# Display state proportions
st.subheader(f"State Proportions (%) Sorted by {sort_by}")
st.dataframe(props_sorted.style.format("{:.2f}"))