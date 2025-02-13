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

def rename_column(col):
    if col.startswith("first_"):
        col = col[len("first_"):]
    elif col.startswith("last_"):
        col = col[len("last_"):]
    col = col.replace("proportion", "%")
    return col

try:
    df = load_data()
    df = df.rename(columns=rename_column)
    state_mapping = {0: 'penetrating', 1: 'oscillating', 2: 'bouncing'}
    st.sidebar.header("Clustering Parameters")
    n_clusters = st.sidebar.slider("Number of clusters", min_value=2, max_value=20, value=10)
    features = st.sidebar.multiselect("Select features for clustering", options=[
        'diameter', 'density', 'mass', 'vm', 'pressure', 'primary_flow', 'particle_feed'
    ], default=[
        'diameter', 'density', 'mass', 'vm', 'pressure', 'primary_flow', 'particle_feed'
    ])
    if len(features) > 0:
        df["state"] = df["state"].map(state_mapping)
        source_penetrating_props = df.groupby("source_file")["state"].apply(lambda x: (x == "penetrating").mean() * 100).reset_index()
        source_penetrating_props.columns = ["source_file", "source_penetrating_%"]
        df = df.merge(source_penetrating_props, on="source_file")
        X = df[features].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        df["cluster"] = kmeans.fit_predict(X_scaled).astype(int)
        cluster_penetrating_props = df.groupby("cluster")["state"].apply(lambda x: (x == "penetrating").mean() * 100).reset_index()
        cluster_penetrating_props.columns = ["cluster", "cluster_penetrating_%"]
        df = df.merge(cluster_penetrating_props, on="cluster")
        exp = hip.Experiment.from_dataframe(df)
        exp.display_data(hip.Displays.PARALLEL_PLOT).update({
            "colorby": "state",
            "height": "800px",
            "order": ["source_file", "source_penetrating_%", "cluster", "cluster_penetrating_%", "state", "diameter", "density", "mass", "vm", "pressure", "primary_flow", "particle_feed"],
            "hide": []
        })
        import streamlit.components.v1 as components
        hip_html = exp.to_html()
        st.header("Interactive Parallel Plot")
        components.html(hip_html, height=800, scrolling=True)
        st.header("Cluster Statistics")
        cols = st.columns(2)
        with cols[0]:
            st.subheader("Overall Statistics")
            total_penetrating = (df["state"] == "penetrating").mean() * 100
            st.write(f"Total penetrating %: {total_penetrating:.1f}%")
            st.write(f"Number of source files: {df['source_file'].nunique()}")
            st.write(f"Total number of particles: {len(df)}")
        with cols[1]:
            st.subheader("Feature Importance")
            feature_importance = pd.DataFrame({
                "Feature": features,
                "Cluster Center Range": [np.ptp(kmeans.cluster_centers_[:, i]) for i in range(len(features))]
            })
            st.dataframe(feature_importance.sort_values("Cluster Center Range", ascending=False))
        st.subheader("Detailed Cluster Statistics")
        for cluster in range(n_clusters):
            with st.expander(f"Cluster {cluster}"):
                cluster_data = df[df["cluster"] == cluster]
                st.write("State distribution:")
                state_counts = cluster_data["state"].value_counts()
                for state_val, count in state_counts.items():
                    st.write(f"  {state_val}: {count} ({count/len(cluster_data)*100:.1f}%)")
                st.write(f"Average source penetrating %: {cluster_data['source_penetrating_%'].mean():.1f}%")
                st.write(f"Cluster penetrating %: {cluster_data['cluster_penetrating_%'].mean():.1f}%")
                source_stats = cluster_data.groupby("source_file")["state"].apply(lambda x: (x == "penetrating").mean() * 100)
                st.write(f"Source file penetrating % range: {source_stats.min():.1f}% - {source_stats.max():.1f}%")
    else:
        st.warning("Please select at least one feature for clustering.")
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
