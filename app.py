import streamlit as st
import hiplot as hip
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering, Birch, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
import umap
import time
import streamlit.components.v1 as components

st.set_page_config(page_title="Clustering Analysis", layout="wide")
st.title("CFD Particle Analysis")

# Utility functions
@st.cache_data
def load_data():
    df = pd.read_parquet('final_states_clean.parquet')
    return df

def rename_column(col):
    if col.startswith("first_"):
        col = col[len("first_"):]
    elif col.startswith("last_"):
        col = col[len("last_"):]
    col = col.replace("proportion", "%")
    return col

def plot_cluster_state_proportions(df):
    """Plot state proportions and numbers for each cluster"""
    # Get state counts and proportions
    cluster_state = df.groupby(['cluster', 'state']).size().unstack(fill_value=0)
    cluster_state_props = cluster_state.div(cluster_state.sum(axis=1), axis=0) * 100
    
    fig = go.Figure()
    
    for state in cluster_state_props.columns:
        # Get counts for this state
        counts = cluster_state[state]
        # Get percentages for this state
        percentages = cluster_state_props[state]
        
        fig.add_trace(go.Bar(
            name=state,
            x=cluster_state_props.index,
            y=percentages,
            text=[f'{count}<br>({pct:.1f}%)' 
                  for count, pct in zip(counts, percentages)],
            textposition='auto',
        ))
    
    # Add total cluster sizes on top of each stack
    total_sizes = cluster_state.sum(axis=1)
    
    fig.add_trace(go.Scatter(
        x=total_sizes.index,
        y=cluster_state_props.sum(axis=1) + 5,  # Add 5% margin
        text=[f'{size}' for size in total_sizes],
        mode='text',
        textposition='top center',
        showlegend=False
        ))  
        
    fig.update_layout(
        barmode='stack',
        title='Cluster Analysis: State Proportions and Counts',
        xaxis_title='Cluster',
        yaxis_title='Proportion (%)',
        height=600,
        showlegend=True,
        legend_title='State'
    )
    
    return fig

def perform_clustering(X_scaled, algorithm, n_clusters):
    """Perform clustering with selected algorithm"""
    if algorithm == "KMeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    elif algorithm == "Spectral":
        clusterer = SpectralClustering(n_clusters=n_clusters, random_state=42, affinity='nearest_neighbors')
    elif algorithm == "BIRCH":
        clusterer = Birch(n_clusters=n_clusters)
    
    return clusterer.fit_predict(X_scaled)

def create_umap_visualization(X_scaled, labels, states, features, penetrating_percentage=None):
    """Create enhanced UMAP visualization with parameter controls"""
    st.subheader("UMAP Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_neighbors = st.slider(
            "Number of neighbors",
            min_value=2,
            max_value=100,
            value=15,
            help="Controls how local/global the embedding is. Lower values capture local structure, higher values capture global structure."
        )
    
    with col2:
        min_dist = st.slider(
            "Minimum distance",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="Controls how tightly points are packed together. Lower values = tighter clusters."
        )
    
    with col3:
        color_by = st.selectbox(
            "Color by",
            options=["Cluster", "State", "Penetrating %"] if penetrating_percentage is not None else ["Cluster", "State"],
            help="Choose how to color the points"
        )

    # # Create UMAP embedding
    # with st.spinner("Generating UMAP projection..."):
    #     reducer = umap.UMAP(
    #         n_neighbors=n_neighbors,
    #         min_dist=min_dist,
    #         random_state=42
    #     )
    #     embedding = reducer.fit_transform(X_scaled)
    
    with st.spinner("Generating UMAP projection..."):
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42,  # Remove random_state to allow parallelism
            n_jobs=-1  # Use all available cores
        )
        embedding = reducer.fit_transform(X_scaled)
        
    # Create dataframe for plotting
    plot_data = pd.DataFrame({
        'UMAP1': embedding[:, 0],
        'UMAP2': embedding[:, 1],
        'Cluster': labels,
        'State': states
    })
    
    if penetrating_percentage is not None:
        plot_data['Penetrating %'] = penetrating_percentage

    # Create the figure without using px
    fig = go.Figure()

    if color_by == "Cluster":
        # Add scatter trace for each cluster
        for cluster in plot_data['Cluster'].unique():
            mask = plot_data['Cluster'] == cluster
            fig.add_trace(go.Scatter(
                x=plot_data.loc[mask, 'UMAP1'],
                y=plot_data.loc[mask, 'UMAP2'],
                mode='markers',
                name=f'Cluster {cluster}',
                marker=dict(size=8),
                text=plot_data.loc[mask, 'State'],
                hovertemplate='State: %{text}<br>UMAP1: %{x}<br>UMAP2: %{y}'
            ))
            
        # Add cluster centers
        cluster_centers = plot_data.groupby('Cluster')[['UMAP1', 'UMAP2']].mean()
        fig.add_trace(go.Scatter(
            x=cluster_centers['UMAP1'],
            y=cluster_centers['UMAP2'],
            mode='markers+text',
            marker=dict(symbol='x', size=15, color='red', line=dict(width=2)),
            text=cluster_centers.index,
            name='Cluster Centers',
            hoverinfo='text'
        ))
            
    elif color_by == "State":
        # Add scatter trace for each state
        for state in plot_data['State'].unique():
            mask = plot_data['State'] == state
            fig.add_trace(go.Scatter(
                x=plot_data.loc[mask, 'UMAP1'],
                y=plot_data.loc[mask, 'UMAP2'],
                mode='markers',
                name=state,
                marker=dict(size=8),
                text=plot_data.loc[mask, 'Cluster'],
                hovertemplate='Cluster: %{text}<br>UMAP1: %{x}<br>UMAP2: %{y}'
            ))
    else:  # Penetrating %
        fig.add_trace(go.Scatter(
            x=plot_data['UMAP1'],
            y=plot_data['UMAP2'],
            mode='markers',
            marker=dict(
                size=8,
                color=plot_data['Penetrating %'],
                colorscale='RdYlBu',
                showscale=True
            ),
            text=[f'State: {s}<br>Cluster: {c}<br>Penetrating: {p:.1f}%' 
                  for s, c, p in zip(plot_data['State'], 
                                   plot_data['Cluster'], 
                                   plot_data['Penetrating %'])],
            hoverinfo='text'
        ))

    fig.update_layout(
        title=f'UMAP Projection (n_neighbors={n_neighbors}, min_dist={min_dist})',
        height=800,
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode='closest'
    )
    
    return fig

def min_max_scale(df, features):
    """Scale features to 0-1 range"""
    scaled_df = df.copy()
    for feature in features:
        min_val = df[feature].min()
        max_val = df[feature].max()
        scaled_df[feature] = (df[feature] - min_val) / (max_val - min_val)
    return scaled_df

# Main app
# Main app
try:
    df = load_data()
    df = df.rename(columns=rename_column)
    state_mapping = {0: 'penetrating', 1: 'oscillating', 2: 'bouncing'}
    
    # Sidebar controls
    st.sidebar.header("Parameters")
    clustering_algorithm = st.sidebar.selectbox(
        "Clustering Algorithm",
        options=["Spectral", "KMeans", "BIRCH"],
        help="Select the clustering algorithm to use"
    )
    
    n_clusters = st.sidebar.slider("Number of clusters", min_value=2, max_value=20, value=10)
    features = st.sidebar.multiselect(
        "Select features for clustering",
        options=['diameter', 'density', 'mass', 'vm', 'pressure', 'primary_flow', 'particle_feed'],
        default=['diameter', 'density', 'mass', 'vm', 'pressure', 'primary_flow', 'particle_feed']
    )
    
    if len(features) > 0:
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Statistical Analysis",
            "Clustering Results",
            "Parallel Plot",
            "UMAP Visualization"
        ])
        
        # Prepare data
        df["state"] = df["state"].map(state_mapping)
        X = df[features].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Tab 1: Statistical Analysis
        with tab1:
            with st.spinner("Analyzing data distributions and correlations..."):
                # Create dummy variables for states
                state_dummies = pd.get_dummies(df['state'], prefix='state')
                
                # Combine features and state dummies
                combined_data = pd.DataFrame(X_scaled, columns=features)
                combined_data = pd.concat([combined_data, state_dummies], axis=1)
                
                # Calculate correlations between features and states only
                feature_state_correlations = pd.DataFrame(
                    np.zeros((len(features), len(state_dummies.columns))),
                    index=features,
                    columns=state_dummies.columns
                )
                
                for feature in features:
                    for state in state_dummies.columns:
                        feature_state_correlations.loc[feature, state] = combined_data[feature].corr(combined_data[state])
                
                # Create correlation heatmap
                fig_corr = go.Figure(data=go.Heatmap(
                    z=feature_state_correlations.values,
                    x=feature_state_correlations.columns,
                    y=feature_state_correlations.index,
                    text=np.round(feature_state_correlations.values, 2),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    hoverongaps=False,
                    colorscale='RdBu',
                    zmid=0
                ))
                
                fig_corr.update_layout(
                    title='Feature-State Correlations',
                    height=800,
                    width=1000,
                    xaxis={'tickangle': 45}
                )
                
                # Display plots
                col1, col2 = st.columns([2, 1])
                
                # State distribution
                state_dist = df['state'].value_counts().reset_index()
                state_dist.columns = ['State', 'Count']
                state_dist['Percentage'] = (state_dist['Count'] / len(df) * 100).round(2)
                
                with col1:
                    st.subheader("Correlations")
                    st.plotly_chart(fig_corr, use_container_width=True)
                
                with col2:
                    st.subheader("Distribution")
                    fig_state = px.pie(
                        state_dist, 
                        values='Count', 
                        names='State',
                        title='Distribution of States'
                    )
                    st.plotly_chart(fig_state, use_container_width=True)
                
                # Average feature values by state
                st.subheader("Average Feature Values by State (Standardized 0-1)")
                scaled_features = min_max_scale(df, features)
                avg_by_state = scaled_features.groupby('state')[features].mean()
                std_by_state = scaled_features.groupby('state')[features].std()

                fig_avg = go.Figure()

                for feature in features:
                    fig_avg.add_trace(go.Bar(
                        name=feature,
                        x=avg_by_state.index,
                        y=avg_by_state[feature],
                        error_y=dict(
                            type='data',
                            array=std_by_state[feature],
                            visible=True
                        )
                    ))

                fig_avg.update_layout(
                    barmode='group',
                    title='Standardized Average Feature Values by State (with standard deviation)',
                    xaxis_title='State',
                    yaxis_title='Standardized Value (0-1)',
                    height=500,
                    yaxis_range=[0, 1]
                )

                st.plotly_chart(fig_avg, use_container_width=True)
        
        # Tab 2: Clustering Results
        with tab2:
            with st.spinner("Performing clustering..."):
                df["cluster"] = perform_clustering(X_scaled, clustering_algorithm, n_clusters)
                
                # Calculate penetrating proportions
                source_penetrating_props = df.groupby("source_file")["state"].apply(
                    lambda x: (x == "penetrating").mean() * 100
                ).reset_index()
                source_penetrating_props.columns = ["source_file", "source_penetrating_%"]
                
                cluster_penetrating_props = df.groupby("cluster")["state"].apply(
                    lambda x: (x == "penetrating").mean() * 100
                ).reset_index()
                cluster_penetrating_props.columns = ["cluster", "cluster_penetrating_%"]
                
                df = df.merge(source_penetrating_props, on="source_file")
                df = df.merge(cluster_penetrating_props, on="cluster")
                
                # Display cluster proportions
                st.header("Cluster State Proportions")
                fig = plot_cluster_state_proportions(df)
                fig.update_layout(
                    barmode='stack',
                    title='Cluster Analysis: State Proportions and Counts',
                    xaxis_title='Cluster',
                    yaxis_title='Proportion (%)',
                    height=600,
                    showlegend=True,
                    legend_title='State',
                    xaxis=dict(dtick=1)
                )

                st.plotly_chart(fig, use_container_width=True)
                
                # Create correlation heatmap between features and clusters
                cluster_dummies = pd.get_dummies(df['cluster'], prefix='cluster')
                feature_cluster_correlations = pd.DataFrame(
                    np.zeros((len(features), len(cluster_dummies.columns))),
                    index=features,
                    columns=cluster_dummies.columns
                )
                
                # Calculate correlations between features and clusters
                for feature in features:
                    for cluster in cluster_dummies.columns:
                        feature_cluster_correlations.loc[feature, cluster] = pd.Series(X_scaled[:, features.index(feature)], index=df.index).corr(cluster_dummies[cluster])

                
                # Create cluster correlation heatmap
                fig_cluster_corr = go.Figure(data=go.Heatmap(
                    z=feature_cluster_correlations.values,
                    x=feature_cluster_correlations.columns,
                    y=feature_cluster_correlations.index,
                    text=np.round(feature_cluster_correlations.values, 2),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    hoverongaps=False,
                    colorscale='RdBu',
                    zmid=0
                ))
                
                fig_cluster_corr.update_layout(
                    title='Feature-Cluster Correlations',
                    height=600,
                    width=1000,
                    xaxis={'tickangle': 45}
                )
                
                
                
                # st.subheader("Feature Importance")
                # if clustering_algorithm == "KMeans":
                #     kmeans = KMeans(n_clusters=n_clusters, n_init="auto")
                #     kmeans.fit(X_scaled)
                #     feature_importance = pd.DataFrame({
                #         "Feature": features,
                #         "Cluster Center Range": [np.ptp(kmeans.cluster_centers_[:, i]) for i in range(len(features))]
                #     })
                #     st.dataframe(feature_importance.sort_values("Cluster Center Range", ascending=False))
                # else:
                #     st.write("Feature importance only available for KMeans clustering")
                
            

                st.subheader("Feature Patterns Across Clusters")
                col1, col2 = st.columns([2, 1])

                with col1:
                    metric = st.selectbox(
                        "Select statistic",
                        options=["Mean", "Median"],
                        help="Choose which central tendency measure to display"
                    )

                with col2:
                    scaling = st.selectbox(
                        "Select scaling",
                        options=["None", "Standard (0-1)"],
                        help="Choose how to scale the feature values"
                    )

                # Prepare the data
                analysis_df = df[features + ['cluster']].copy()

                if scaling == "Standard (0-1)":
                    # Scale features to 0-1 range
                    for feature in features:
                        min_val = analysis_df[feature].min()
                        max_val = analysis_df[feature].max()
                        analysis_df[feature] = (analysis_df[feature] - min_val) / (max_val - min_val)

                # Calculate statistics
                if metric == "Mean":
                    cluster_stats = analysis_df.groupby('cluster')[features].mean()
                    cluster_ranges = analysis_df.groupby('cluster')[features].agg(lambda x: x.max() - x.min())
                else:  # Median
                    cluster_stats = analysis_df.groupby('cluster')[features].median()
                    cluster_ranges = analysis_df.groupby('cluster')[features].agg(lambda x: x.max() - x.min())

                # Transpose the dataframes
                cluster_stats = cluster_stats.T
                cluster_ranges = cluster_ranges.T

                # Create central tendency heatmap
                fig_stats = go.Figure(data=go.Heatmap(
                    z=cluster_stats.values,
                    x=cluster_stats.columns,  # These are now the clusters
                    y=cluster_stats.index,    # These are now the features
                    text=np.round(cluster_stats.values, 2),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    hoverongaps=False,
                    colorscale='RdBu',
                    zmid=cluster_stats.values.mean()
                ))

                fig_stats.update_layout(
                    title=f'Feature {metric} Values by Cluster{" (Standardized)" if scaling == "Standard (0-1)" else ""}',
                    xaxis_title='Cluster',
                    yaxis_title='Feature',
                    height=600,
                    xaxis=dict(
                        tickmode='linear',  # Show all cluster numbers
                        dtick=1            # Space between ticks
                    )
                )

                # Display plots
                st.plotly_chart(fig_stats, use_container_width=True)

                st.subheader("Feature-Cluster Correlations")
                st.plotly_chart(fig_cluster_corr, use_container_width=True)
                    
        # Tab 3: Parallel Plot
        with tab3:
            with st.spinner("Generating parallel plot..."):
                st.header("Interactive Parallel Plot")
                exp = hip.Experiment.from_dataframe(df)
                exp.display_data(hip.Displays.PARALLEL_PLOT).update({
                    "colorby": "state",
                    "height": "800px",
                    "order": ["source_file", "source_penetrating_%", "cluster", "cluster_penetrating_%", "state"] + features,
                    "hide": []
                })
                hip_html = exp.to_html()
                components.html(hip_html, height=800, scrolling=True)
        
        # Tab 4: UMAP Visualization
        with tab4:
            st.header("UMAP Visualization")
            
            fig = create_umap_visualization(
                X_scaled,
                df['cluster'],
                df['state'],
                features,
                df['source_penetrating_%']
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("How to Interpret UMAP Visualization"):
                st.write("""
                ### UMAP Visualization Guide
                
                1. **Clusters**: Points close together suggest similar particles
                2. **Parameter Effects**:
                - Higher n_neighbors: More emphasis on global structure
                - Lower n_neighbors: More emphasis on local structure
                - Higher min_dist: More spread out points
                - Lower min_dist: Tighter clusters
                
                3. **Color Schemes**:
                - Cluster view: See how algorithm grouped particles
                - State view: Compare algorithmic clusters with actual states
                - Penetrating % view: See penetration patterns
                
                4. **What to Look For**:
                - Well-separated clusters suggest distinct particle behaviors
                - Overlapping clusters might indicate similar behaviors
                - Gradients in penetrating % can show transition zones
                """)

    else:
        st.warning("Please select at least one feature for clustering.")

except Exception as e:
    st.error(f"Error: {str(e)}")