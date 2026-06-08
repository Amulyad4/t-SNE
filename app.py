import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# Page config
st.set_page_config(page_title="t-SNE Visualization", page_icon="📊", layout="wide")

st.title("📊 t-SNE Visualization")

# Upload dataset
uploaded = st.file_uploader("Upload CSV/Excel File", type=["csv", "xlsx", "xls"])

if uploaded:

    # Read file
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    st.success(f"Dataset Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    # Numeric columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(num_cols) < 2:
        st.error("Need at least 2 numeric columns")
        st.stop()

    # Feature selection
    st.subheader("Select Features")

    feature_cols = st.multiselect(
        "Choose numeric columns",
        options=num_cols,
        default=num_cols
    )

    if len(feature_cols) < 2:
        st.warning("Select at least 2 columns")
        st.stop()

    X = df[feature_cols].dropna()

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # t-SNE Parameters
    st.subheader("t-SNE Settings")

    perplexity = st.slider("Perplexity", 5, 50, 30)
    learning_rate = st.slider("Learning Rate", 10, 1000, 200)
    iterations = st.slider("Iterations", 250, 2000, 1000)

    st.divider()

    # ==========================
    # Visualization 1
    # 2D t-SNE
    # ==========================
    st.subheader("1️⃣ 2D t-SNE Scatter Plot")

    tsne_2d = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=iterations,
        random_state=42
    )

    X_2d = tsne_2d.fit_transform(X_scaled)

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.7)
    ax1.set_xlabel("Dimension 1")
    ax1.set_ylabel("Dimension 2")
    ax1.set_title("2D t-SNE Projection")

    st.pyplot(fig1)

    # ==========================
    # Visualization 2
    # 3D t-SNE
    # ==========================
    if len(feature_cols) >= 3:

        st.subheader("2️⃣ Interactive 3D t-SNE")

        tsne_3d = TSNE(
            n_components=3,
            perplexity=perplexity,
            learning_rate=learning_rate,
            max_iter=iterations,
            random_state=42
        )

        X_3d = tsne_3d.fit_transform(X_scaled)

        fig2 = px.scatter_3d(
            x=X_3d[:, 0],
            y=X_3d[:, 1],
            z=X_3d[:, 2],
            labels={
                "x": "Dim 1",
                "y": "Dim 2",
                "z": "Dim 3"
            },
            title="3D t-SNE Projection"
        )

        st.plotly_chart(fig2, use_container_width=True)

    # ==========================
    # Visualization 3
    # Data Distribution
    # ==========================
    st.subheader("3️⃣ Data Distribution")

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.hist(X_scaled.flatten(), bins=30)
    ax3.set_title("Feature Distribution")
    st.pyplot(fig3)