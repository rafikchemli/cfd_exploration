#!/usr/bin/env python3
"""
CFD Figure Export Tool

This script exports publication-quality figures from the CFD analysis data.
It follows journal publication guidelines for figure format, resolution,
and appearance.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering, Birch
import umap
from pathlib import Path

# Set up accessible color schemes
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

# Create colorblind-friendly palettes
COLORBLIND_PALETTE = sns.color_palette("colorblind", 10)
sns.set_palette(COLORBLIND_PALETTE)

# Journal-specific sizing (single column and full width options)
SINGLE_COLUMN_SIZE = (3.5, 3.5)  # in inches (roughly corresponds to 1063 pixels at 300 dpi)
FULL_WIDTH_SIZE = (7.5, 5)  # in inches (roughly corresponds to 2244 pixels at 300 dpi)

def load_data(data_path):
    """Load and prepare the CFD data.
    
    Args:
        data_path: Path to the parquet data file
        
    Returns:
        Processed DataFrame
    """
    df = pd.read_parquet(data_path)
    
    # Rename columns for clarity
    def rename_column(col):
        if col.startswith("first_"):
            col = col[len("first_"):]
        elif col.startswith("last_"):
            col = col[len("last_"):]
        col = col.replace("proportion", "%")
        return col
    
    df = df.rename(columns=rename_column)
    
    # Map state values to readable names
    state_mapping = {0: 'penetrating', 1: 'oscillating', 2: 'bouncing'}
    df["state"] = df["state"].map(state_mapping)
    
    return df

def perform_clustering(X_scaled, algorithm, n_clusters):
    """Perform clustering with selected algorithm.
    
    Args:
        X_scaled: Scaled feature data
        algorithm: Clustering algorithm name
        n_clusters: Number of clusters
        
    Returns:
        Cluster labels
    """
    if algorithm == "KMeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")  # Match app.py
    elif algorithm == "Spectral":
        clusterer = SpectralClustering(n_clusters=n_clusters, random_state=42, 
                                      affinity='nearest_neighbors')  # Match app.py
    elif algorithm == "BIRCH":
        clusterer = Birch(n_clusters=n_clusters)  # BIRCH doesn't take random_state
    
    print(f"Performing {algorithm} clustering with n_clusters={n_clusters}, random_state=42")
    return clusterer.fit_predict(X_scaled)

def export_state_distribution(df, output_dir, width='single', figure_num=1):
    """Create and export pie chart of state distribution.
    
    Args:
        df: Data DataFrame
        output_dir: Directory to save figure
        width: 'single' for single column or 'full' for full page width
        figure_num: Figure number for caption
    """
    fig_size = SINGLE_COLUMN_SIZE if width == 'single' else FULL_WIDTH_SIZE
    
    state_counts = df['state'].value_counts()
    state_percentages = (state_counts / len(df) * 100).round(1)
    
    fig, ax = plt.subplots(figsize=fig_size)
    wedges, texts, autotexts = ax.pie(
        state_counts, 
        labels=state_counts.index,
        autopct='%1.1f%%',
        colors=[COLORBLIND_PALETTE[i] for i in range(len(state_counts))],
        startangle=90,
        wedgeprops={'edgecolor': 'w', 'linewidth': 1}
    )
    
    # Styling
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title('Distribution of Particle States')
    plt.tight_layout()
    
    # Save only in PNG format
    fmt = 'png'
    dpi = 1000
    filename = os.path.join(output_dir, f"Figure_1_State_Distribution.{fmt}")
    plt.savefig(filename, format=fmt, dpi=dpi, bbox_inches='tight')
    
    plt.close()
    
    # Create caption file
    caption_file = os.path.join(output_dir, "figure_captions.txt")
    with open(caption_file, 'w' if figure_num == 1 else 'a') as f:
        if figure_num == 1:
            f.write("Figure Captions\n==============\n\n")
        f.write(f"Figure {figure_num}: Distribution of particle states in the CFD simulation. "
                f"The dataset contains {len(df)} particles distributed across "
                f"{len(state_counts)} states ({', '.join(state_counts.index)}).\n\n")

def export_feature_correlations(df, features, output_dir, width='single', feature_labels=None, figure_num=2):
    """Create and export heatmap of feature correlations with states.
    
    Args:
        df: Data DataFrame
        features: List of feature names
        output_dir: Directory to save figure
        width: 'single' for single column or 'full' for full page width
        feature_labels: Optional custom labels for features
        figure_num: Figure number for caption
    """
    # Use custom labels if provided
    display_features = feature_labels if feature_labels is not None else features
    fig_size = SINGLE_COLUMN_SIZE if width == 'single' else FULL_WIDTH_SIZE
    
    # Create dummy variables for states
    state_dummies = pd.get_dummies(df['state'], prefix='state')
    
    # Standardize features
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Combine features and state dummies
    combined_data = pd.DataFrame(X_scaled, columns=features)
    combined_data = pd.concat([combined_data, state_dummies], axis=1)
    
    # Calculate correlations
    feature_state_correlations = pd.DataFrame(
        np.zeros((len(features), len(state_dummies.columns))),
        index=features,
        columns=state_dummies.columns
    )
    
    for feature in features:
        for state in state_dummies.columns:
            feature_state_correlations.loc[feature, state] = combined_data[feature].corr(combined_data[state])
    
    # Create correlation heatmap
    fig, ax = plt.subplots(figsize=fig_size)
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    sns.heatmap(
        feature_state_correlations,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .8},
        ax=ax
    )
    
    # Use custom labels if provided, making sure lengths match
    if feature_labels is not None:
        # Ensure the number of labels matches the number of features
        if len(feature_labels) == len(feature_state_correlations.index):
            ax.set_yticklabels(feature_labels, rotation=0)  # Horizontal labels
        else:
            print(f"Warning: Number of labels ({len(feature_labels)}) doesn't match features ({len(feature_state_correlations.index)})")
            print(f"Features: {feature_state_correlations.index.tolist()}")
            print(f"Labels: {feature_labels}")
        
    ax.set_title('Feature-State Correlations')
    plt.tight_layout()
    
    # Save only in PNG format
    fmt = 'png'
    dpi = 1000
    filename = os.path.join(output_dir, f"Figure_2_Feature_State_Correlations.{fmt}")
    plt.savefig(filename, format=fmt, dpi=dpi, bbox_inches='tight')
    
    plt.close()
    
    # Append caption
    caption_file = os.path.join(output_dir, "figure_captions.txt")
    with open(caption_file, 'a') as f:
        f.write(f"Figure {figure_num}: Correlation heatmap between particle features and states. "
                "Positive values (red) indicate positive correlation, while negative values (blue) "
                "indicate inverse correlation. The analysis reveals which features are most "
                "predictive of specific particle states.\n\n")

def export_cluster_state_proportions(df, output_dir, width='full', feature_labels=None, figure_num=3):
    """Create and export stacked bar chart of cluster state proportions.
    
    Args:
        df: Data DataFrame with cluster assignments
        output_dir: Directory to save figure
        width: 'single' for single column or 'full' for full page width
        feature_labels: Optional custom labels for features
        figure_num: Figure number for caption
    """
    fig_size = SINGLE_COLUMN_SIZE if width == 'single' else FULL_WIDTH_SIZE
    
    # Get state counts and proportions by cluster
    cluster_state = df.groupby(['cluster', 'state']).size().unstack(fill_value=0)
    cluster_state_props = cluster_state.div(cluster_state.sum(axis=1), axis=0) * 100
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Create stacked bars
    bottom = np.zeros(len(cluster_state))
    for state in cluster_state.columns:
        values = cluster_state_props[state].values
        ax.bar(
            cluster_state.index,
            values,
            bottom=bottom,
            label=state,
            edgecolor='white',
            linewidth=1
        )
        
        # Add percentage labels in the middle of each segment
        for i, (v, b) in enumerate(zip(values, bottom)):
            if v > 5:  # Only add text if segment is large enough
                ax.text(
                    i,
                    b + v/2,
                    f'{v:.1f}%',
                    ha='center',
                    va='center',
                    fontsize=8,
                    color='white',
                    fontweight='bold'
                )
                
        bottom += values
    
    # Add total cluster sizes on top of each stack
    total_sizes = cluster_state.sum(axis=1)
    for i, total in enumerate(total_sizes):
        ax.text(
            i,
            105,  # Position slightly above the bar
            f'n={total}',
            ha='center',
            fontsize=9
        )
    
    # Styling
    ax.set_title('Cluster Analysis: State Proportions and Counts')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Proportion (%)')
    ax.set_ylim(0, 110)  # Make room for labels
    ax.set_xticks(range(len(cluster_state)))
    ax.set_xticklabels(cluster_state.index)
    
    # Add legend outside the plot on the right
    ax.legend(title='State', loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout(pad=1.2)  # Add padding for the legend
    
    # Save only in PNG format
    fmt = 'png'
    dpi = 1000
    filename = os.path.join(output_dir, f"Figure_3_Cluster_State_Proportions.{fmt}")
    plt.savefig(filename, format=fmt, dpi=dpi, bbox_inches='tight')
    
    plt.close()
    
    # Append caption
    caption_file = os.path.join(output_dir, "figure_captions.txt")
    with open(caption_file, 'a') as f:
        f.write(f"Figure {figure_num}: Proportion of particle states within each cluster. "
                f"The stacked bar chart shows how particles with different states "
                f"({', '.join(cluster_state.columns)}) are distributed across the "
                f"{len(cluster_state)} identified clusters. Numbers at the top indicate "
                "the total count of particles in each cluster.\n\n")

def export_umap_visualization(df, features, output_dir, width='full', feature_labels=None, figure_num=4):
    """Create and export UMAP visualization colored by state.
    
    Args:
        df: Data DataFrame with cluster assignments
        features: List of feature names
        output_dir: Directory to save figure
        width: 'single' for single column or 'full' for full page width
        feature_labels: Optional custom labels for features
        figure_num: Figure number for caption
    """
    fig_size = SINGLE_COLUMN_SIZE if width == 'single' else FULL_WIDTH_SIZE
    
    # Standardize features
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create UMAP embedding with fixed random state for reproducibility
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        random_state=42,
        n_jobs=-1
    )
    
    embedding = reducer.fit_transform(X_scaled)
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Get unique states and assign colors
    states = df['state'].unique()
    
    # Create scatter plot for each state
    for i, state in enumerate(states):
        mask = df['state'] == state
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=30,
            alpha=0.7,
            label=state,
            edgecolors='none'
        )
    
    # Styling
    ax.set_title('UMAP Projection of Particle Features')
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    
    # Add legend outside the plot on the right
    ax.legend(title='State', markerscale=1.5, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout(pad=1.2)  # Add padding for the legend
    
    # Save only in PNG format
    fmt = 'png'
    dpi = 1000
    filename = os.path.join(output_dir, f"Figure_4_UMAP_Visualization.{fmt}")
    plt.savefig(filename, format=fmt, dpi=dpi, bbox_inches='tight')
    
    plt.close()
    
    # Append caption
    caption_file = os.path.join(output_dir, "figure_captions.txt")
    with open(caption_file, 'a') as f:
        f.write(f"Figure {figure_num}: UMAP dimensionality reduction visualization of the particle feature space. "
                "Points are colored by particle state. The proximity of points in the projection "
                "indicates similarity in the multi-dimensional feature space, revealing natural "
                "clustering patterns in the data. This visualization uses a 15-nearest neighbor "
                "approach with a minimum distance of 0.1.\n\n")

def export_feature_distributions(df, features, output_dir, width='full', feature_labels=None, figure_num=5):
    """Create and export feature distributions by state.
    
    Args:
        df: Data DataFrame
        features: List of feature names
        output_dir: Directory to save figure
        width: 'single' for single column or 'full' for full page width
        feature_labels: Optional custom labels for features
        figure_num: Figure number for caption
    """
    # Use custom labels if provided
    display_features = feature_labels if feature_labels is not None else features
    fig_size = SINGLE_COLUMN_SIZE if width == 'single' else FULL_WIDTH_SIZE
    
    # Scaling features to 0-1 range
    scaled_df = df.copy()
    for feature in features:
        min_val = df[feature].min()
        max_val = df[feature].max()
        scaled_df[feature] = (df[feature] - min_val) / (max_val - min_val)
    
    # Calculate means and std devs
    avg_by_state = scaled_df.groupby('state')[features].mean()
    std_by_state = scaled_df.groupby('state')[features].std()
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Set bar width
    n_features = len(features)
    n_states = len(avg_by_state)
    width_bar = 0.8 / n_features
    
    # Create grouped bars
    for i, feature in enumerate(features):
        positions = np.arange(n_states) + (i - n_features/2 + 0.5) * width_bar
        
        bars = ax.bar(
            positions,
            avg_by_state[feature],
            width=width_bar,
            yerr=std_by_state[feature],
            label=feature,
            capsize=3
        )
    
    # Styling
    ax.set_title('Standardized Feature Values by State')
    ax.set_ylabel('Standardized Value (0-1)')
    ax.set_ylim(0, 1)
    ax.set_xticks(range(n_states))
    ax.set_xticklabels(avg_by_state.index)
    
    # Add legend with custom labels if provided outside the plot on the right
    if feature_labels is not None:
        # Ensure the number of labels matches the number of features
        handles, _ = ax.get_legend_handles_labels()
        if len(feature_labels) == len(handles):
            ax.legend(handles, feature_labels, title='Feature', loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            print(f"Warning: Number of labels ({len(feature_labels)}) doesn't match number of bars ({len(handles)})")
            print(f"Features from handles: {len(handles)}")
            print(f"Labels: {feature_labels}")
            ax.legend(title='Feature', loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        ax.legend(title='Feature', loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout(pad=1.2)  # Add padding for the legend
    
    # Save only in PNG format
    fmt = 'png'
    dpi = 1000
    filename = os.path.join(output_dir, f"Figure_5_Feature_Distributions.{fmt}")
    plt.savefig(filename, format=fmt, dpi=dpi, bbox_inches='tight')
    
    plt.close()
    
    # Append caption
    caption_file = os.path.join(output_dir, "figure_captions.txt")
    with open(caption_file, 'a') as f:
        f.write(f"Figure {figure_num}: Standardized feature values by particle state. "
                "Bars show the mean value for each feature (scaled to 0-1 range) "
                "across the different particle states, with error bars representing "
                "one standard deviation. This visualization highlights which features "
                "are most distinctly associated with each state.\n\n")

def export_feature_patterns(df, features, output_dir, width='full', feature_labels=None, figure_num=6):
    """Create and export heatmap of feature patterns across clusters.
    
    Args:
        df: Data DataFrame with cluster assignments
        features: List of feature names
        output_dir: Directory to save figure
        width: 'single' for single column or 'full' for full page width
        feature_labels: Optional custom labels for features
        figure_num: Figure number for caption
    """
    # Use custom labels if provided
    display_features = feature_labels if feature_labels is not None else features
    fig_size = SINGLE_COLUMN_SIZE if width == 'single' else FULL_WIDTH_SIZE
    
    # Calculate mean values for each feature by cluster
    cluster_stats = df.groupby('cluster')[features].mean()
    
    # Standardize the data for visualization
    cluster_stats_std = cluster_stats.copy()
    for feature in features:
        min_val = cluster_stats[feature].min()
        max_val = cluster_stats[feature].max()
        cluster_stats_std[feature] = (cluster_stats[feature] - min_val) / (max_val - min_val)
    
    # Transpose the dataframe for heatmap
    cluster_stats_std = cluster_stats_std.T
    
    # Create the figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Create the heatmap
    sns.heatmap(
        cluster_stats_std,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        linewidths=.5,
        cbar_kws={"shrink": .8},
        ax=ax
    )
    
    # Use custom labels if provided, making sure lengths match
    if feature_labels is not None:
        # Ensure the number of labels matches the number of features
        if len(feature_labels) == len(cluster_stats_std.index):
            ax.set_yticklabels(feature_labels, rotation=0)  # Horizontal labels
        else:
            print(f"Warning: Number of labels ({len(feature_labels)}) doesn't match features ({len(cluster_stats_std.index)})")
            print(f"Features: {cluster_stats_std.index.tolist()}")
            print(f"Labels: {feature_labels}")
    
    # Styling
    ax.set_title('Feature Patterns Across Clusters (Standardized)')
    ax.set_xlabel('Cluster')
    plt.tight_layout(pad=1.2)  # Add padding for the legend
    
    # Save only in PNG format
    fmt = 'png'
    dpi = 1000
    filename = os.path.join(output_dir, f"Figure_6_Feature_Patterns.{fmt}")
    plt.savefig(filename, format=fmt, dpi=dpi, bbox_inches='tight')
    
    plt.close()
    
    # Append caption
    caption_file = os.path.join(output_dir, "figure_captions.txt")
    with open(caption_file, 'a') as f:
        f.write(f"Figure {figure_num}: Feature patterns across clusters. "
                "The heatmap shows the standardized mean values for each feature across different clusters. "
                "This visualization highlights which features characterize each cluster and reveals "
                "potential patterns in the multi-dimensional feature space.\n\n")

def export_parallel_plot(df, features, output_dir, width='full', feature_labels=None, figure_num=7):
    """Create and export parallel coordinates plot.
    
    Args:
        df: Data DataFrame with cluster assignments
        features: List of feature names
        output_dir: Directory to save figure
        width: 'single' for single column or 'full' for full page width
        feature_labels: Optional custom labels for features
        figure_num: Figure number for caption
    """
    # Use custom labels if provided
    display_features = feature_labels if feature_labels is not None else features
    fig_size = SINGLE_COLUMN_SIZE if width == 'single' else FULL_WIDTH_SIZE
    
    # Scale features to 0-1 range for better visualization
    scaled_df = df.copy()
    for feature in features:
        min_val = df[feature].min()
        max_val = df[feature].max()
        scaled_df[feature] = (df[feature] - min_val) / (max_val - min_val)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(fig_size[0] * 1.5, fig_size[1]))
    
    # Get unique states and assign colors
    states = df['state'].unique()
    color_dict = {state: COLORBLIND_PALETTE[i] for i, state in enumerate(states)}
    
    # Plot each state with a different color
    for state in states:
        state_data = scaled_df[scaled_df['state'] == state]
        
        # For each data point
        for i, row in state_data.iterrows():
            # Get x and y coordinates
            xs = np.arange(len(features))
            ys = [row[feature] for feature in features]
            
            # Plot the line
            ax.plot(xs, ys, alpha=0.4, c=color_dict[state], linewidth=1)
    
    # Add state means
    for state in states:
        state_means = scaled_df[scaled_df['state'] == state][features].mean()
        ys = [state_means[feature] for feature in features]
        ax.plot(xs, ys, c=color_dict[state], linewidth=3, label=state)
    
    # Styling
    ax.set_xticks(np.arange(len(features)))
    # Use custom labels if provided
    if feature_labels is not None:
        ax.set_xticklabels(feature_labels, rotation=45, ha='right')
    else:
        ax.set_xticklabels(features, rotation=45, ha='right')
    
    ax.set_title('Parallel Coordinates Plot of Features by State')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Standardized Value (0-1)')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend outside the plot on the right
    ax.legend(title='State', loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout(pad=1.2)  # Add padding for the legend
    
    # Save only in PNG format
    fmt = 'png'
    dpi = 1000
    filename = os.path.join(output_dir, f"Figure_7_Parallel_Plot.{fmt}")
    plt.savefig(filename, format=fmt, dpi=dpi, bbox_inches='tight')
    
    plt.close()
    
    # Append caption
    caption_file = os.path.join(output_dir, "figure_captions.txt")
    with open(caption_file, 'a') as f:
        f.write(f"Figure {figure_num}: Parallel coordinates plot of standardized feature values by state. "
                "Each line represents a particle, colored by its state, with the bold lines showing "
                "the mean values for each state. This visualization allows for the comparison of "
                "multi-dimensional feature distributions across different particle states.\n\n")

def main():
    """Main function to parse arguments and execute figure export."""
    parser = argparse.ArgumentParser(description='Export journal-quality figures from CFD data')
    
    parser.add_argument('--data', type=str, default='final_states_clean.parquet',
                      help='Path to the parquet data file')
    
    parser.add_argument('--output', type=str, default='figures',
                      help='Directory to save output figures')
    
    parser.add_argument('--features', type=str, nargs='+',
                      default=['diameter', 'density', 'mass', 'vm', 'pressure', 'primary_flow', 'particle_feed'],
                      help='Feature columns to use for analysis')
                      
    parser.add_argument('--labels', type=str, nargs='+',
                      default=None,
                      help='Custom labels for features (same order as features)')
    
    parser.add_argument('--clustering', type=str, default='Spectral',
                      choices=['KMeans', 'Spectral', 'BIRCH'],
                      help='Clustering algorithm to use')
    
    parser.add_argument('--n_clusters', type=int, default=6,
                      help='Number of clusters to create')
    
    parser.add_argument('--width', type=str, default='full',
                      choices=['single', 'full'],
                      help='Figure width: single column or full page width')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {args.data}")
    df = load_data(args.data)
    
    # Perform clustering if needed
    X = df[args.features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Clustering is handled by the perform_clustering function now
    print(f"Using features for clustering: {args.features}")
    df['cluster'] = perform_clustering(X_scaled, args.clustering, args.n_clusters)
    print("Cluster distribution:", df['cluster'].value_counts().sort_index().to_dict())
    
    # Create a fresh caption file
    caption_file = os.path.join(output_dir, "figure_captions.txt")
    with open(caption_file, 'w') as f:
        f.write("Figure Captions\n==============\n\n")
    
    # Export figures
    print("Exporting figures...")
    export_state_distribution(df, output_dir, args.width, figure_num=1)
    print("✓ Exported Figure 1: State Distribution")
    
    export_feature_correlations(df, args.features, output_dir, args.width, args.labels, figure_num=2)
    print("✓ Exported Figure 2: Feature-State Correlations")
    
    export_cluster_state_proportions(df, output_dir, args.width, figure_num=3)
    print("✓ Exported Figure 3: Cluster State Proportions")
    
    export_umap_visualization(df, args.features, output_dir, args.width, figure_num=4)
    print("✓ Exported Figure 4: UMAP Visualization")
    
    export_feature_distributions(df, args.features, output_dir, args.width, args.labels, figure_num=5)
    print("✓ Exported Figure 5: Feature Distributions")
    
    export_feature_patterns(df, args.features, output_dir, args.width, args.labels, figure_num=6)
    print("✓ Exported Figure 6: Feature Patterns Across Clusters")
    
    export_parallel_plot(df, args.features, output_dir, args.width, args.labels, figure_num=7)
    print("✓ Exported Figure 7: Parallel Coordinates Plot")
    
    print(f"\nAll figures exported to {output_dir}")
    print(f"Figure captions saved to {os.path.join(output_dir, 'figure_captions.txt')}")

if __name__ == "__main__":
    main()
    