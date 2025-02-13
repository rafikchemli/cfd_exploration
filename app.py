import streamlit as st
import hiplot as hip
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Clustering Analysis", layout="wide")
st.title("Particle States Clustering Analysis")

@st.cache_data
def load_data():
    df = pd.read_parquet('final_states_clean.parquet')
    return df

try:
    df = load_data()
    state_mapping = {0: 'penetrating', 1: 'oscillating', 2: 'bouncing'}
    st.sidebar.header("Clustering Parameters")
    n_clusters = st.sidebar.slider("Number of clusters", min_value=2, max_value=20, value=10)
    features = st.sidebar.multiselect("Select features for clustering", options=[
        'first_diameter', 'first_density', 'first_mass', 'first_vm', 'first_pressure', 'first_primary_flow', 'first_particle_feed'
    ], default=[
        'first_diameter', 'first_density', 'first_mass', 'first_vm', 'first_pressure', 'first_primary_flow', 'first_particle_feed'
    ])
    if len(features) > 0:
        df['last_state_name'] = df['last_state'].map(state_mapping)
        source_penetrating_props = df.groupby('first_source_file')['last_state'].apply(lambda x: (x == 0).mean() * 100).reset_index()
        source_penetrating_props.columns = ['first_source_file', 'source_penetrating_proportion']
        df = df.merge(source_penetrating_props, on='first_source_file')
        X = df[features].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        df['cluster'] = kmeans.fit_predict(X_scaled).astype(int)
        cluster_penetrating_props = df.groupby('cluster')['last_state'].apply(lambda x: (x == 0).mean() * 100).reset_index()
        cluster_penetrating_props.columns = ['cluster', 'cluster_penetrating_proportion']
        df = df.merge(cluster_penetrating_props, on='cluster')
        exp = hip.Experiment.from_dataframe(df)
        exp.display_data(hip.Displays.PARALLEL_PLOT).update({
            "colorby": "last_state_name",
            "height": "800px",
            "order": ["first_source_file", "source_penetrating_proportion", "cluster", "cluster_penetrating_proportion", "last_state_name", "first_diameter", "first_density", "first_mass", "first_vm", "first_pressure", "first_primary_flow", "first_particle_feed"],
            "hide": ["last_state"]
        })
        import streamlit.components.v1 as components
        hip_html = exp.to_html()
        st.header("Interactive Parallel Plot")
        components.html(hip_html, height=800, scrolling=True)
        st.header("Cluster Statistics")
        cols = st.columns(2)
        with cols[0]:
            st.subheader("Overall Statistics")
            total_penetrating = (df['last_state'] == 0).mean() * 100
            st.write(f"Total penetrating proportion: {total_penetrating:.1f}%")
            st.write(f"Number of source files: {df['first_source_file'].nunique()}")
            st.write(f"Total number of particles: {len(df)}")
        with cols[1]:
            st.subheader("Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': features,
                'Cluster Center Range': [np.ptp(kmeans.cluster_centers_[:, i]) for i in range(len(features))]
            })
            st.dataframe(feature_importance.sort_values('Cluster Center Range', ascending=False))
        st.subheader("Detailed Cluster Statistics")
        for cluster in range(n_clusters):
            with st.expander(f"Cluster {cluster}"):
                cluster_data = df[df['cluster'] == cluster]
                st.write("State distribution:")
                state_counts = cluster_data['last_state_name'].value_counts()
                for state, count in state_counts.items():
                    st.write(f"  {state}: {count} ({count/len(cluster_data)*100:.1f}%)")
                st.write(f"Average source penetrating proportion: {cluster_data['source_penetrating_proportion'].mean():.1f}%")
                st.write(f"Cluster penetrating proportion: {cluster_data['cluster_penetrating_proportion'].mean():.1f}%")
                source_stats = cluster_data.groupby('first_source_file')['last_state'].apply(lambda x: (x == 0).mean() * 100)
                st.write(f"Source file penetrating proportions range: {source_stats.min():.1f}% - {source_stats.max():.1f}%")
    else:
        st.warning("Please select at least one feature for clustering.")
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
