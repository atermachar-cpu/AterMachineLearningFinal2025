import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px


def load_data():
    return pd.read_csv("dirty_cafe_sales.csv")

df = load_data()
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Encode categorical features automatically
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))


feature_list = df.select_dtypes(include=[np.number]).columns.tolist()
x_feature = st.sidebar.selectbox("Select X-axis feature", feature_list)
y_feature = st.sidebar.selectbox("Select Y-axis feature", feature_list)
k = st.sidebar.slider("Number of clusters (k)", 2, 10, 3)
cluster_choice = st.sidebar.selectbox("Select cluster to view", list(range(k)))

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[feature_list])

# KMeans clustering
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(scaled_features)
df["cluster"] = labels

# Dashboard title
st.title("Cafe Sales K-Means Clustering Dashboard")

# Layout
col_table, col_chart = st.columns([1,2])

with col_table:
    st.subheader("Cluster Data Table")
    st.write(df[df["cluster"] == cluster_choice])
    if st.checkbox("Show full dataset"):
        st.write(df)

with col_chart:
    st.subheader("Cluster Visualization")
    centroids_scaled = kmeans.cluster_centers_
    centroids = scaler.inverse_transform(centroids_scaled)
    centroid_df = pd.DataFrame(centroids, columns=feature_list)
    centroid_df["cluster"] = range(k)

    fig = px.scatter(
        df, x=x_feature, y=y_feature, color="cluster",
        title="Cluster Visualization", opacity=0.7
    )
    fig.add_scatter(
        x=centroid_df[x_feature], y=centroid_df[y_feature],
        mode="markers", marker=dict(size=15, symbol="x", color="black"),
        name="Centroids"
    )
    st.plotly_chart(fig, use_container_width=True)

# Prediction section
st.markdown("---")
st.header("Predict Cluster for New Cafe Sale")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)


with col1:
    item_choice = st.selectbox("Item", df["Item"].unique())
with col2:
    quantity = st.number_input("Quantity", min_value=1, max_value=100, value=3, step=1)
with col3:
    price_per_unit = st.number_input("Price Per Unit", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
with col4:
    total_spent = st.number_input("Total Spent", min_value=0.0, max_value=1000.0, value=20.0, step=1.0)


# Encode new input consistently
item_encoded = df.loc[df["Item"] == item_choice, "Item"].iloc[0]


if st.button("Predict Cluster"):
    new_point = np.array([[quantity, price_per_unit, total_spent, item_encoded]])
    new_point_scaled = scaler.transform(new_point)
    pred_cluster = kmeans.predict(new_point_scaled)[0]
    st.success(f"This new cafe sale belongs to **Cluster {pred_cluster}**")
