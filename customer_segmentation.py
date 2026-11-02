import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
print("PROGRAM STARTED")
print("Customer Segmentation using K-Means")

# Create sample customer dataset

data = {
    "CustomerID": range(1, 201),
    "Age": np.random.randint(18, 65, 200),
    "Annual Income (k$)": np.random.randint(15, 140, 200),
    "Spending Score (1-100)": np.random.randint(1, 100, 200)
}

df = pd.DataFrame(data)

print("\nDataset loaded successfully!")
print(df.head())

print("\nDataset Shape:", df.shape)
print("\nDataset Information:")
print(df.info())
# Select features for clustering
X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]

print("\nSelected Features:")
print(X.head())

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nData Standardized Successfully!")
print("Scaled Data Shape:", X_scaled.shape)
# Elbow Method
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)

plt.savefig("elbow_method.png")
plt.close()

print("\nElbow Method completed!")
# Apply K-Means Clustering

print("\nStarting K-Means Clustering...")

# Create K-Means Model
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

# Train Model and Assign Clusters
df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nK-Means Clustering Completed!")

# Display Customer Groups
print("\nCustomer Cluster Groups:")

print(
    df[["CustomerID", "Age", "Annual Income (k$)",
        "Spending Score (1-100)", "Cluster"]].head(10)
)

# Count Customers in Each Cluster
print("\nNumber of Customers in Each Cluster:")

print(df["Cluster"].value_counts().sort_index())

# Save Results
df.to_csv("customer_segments.csv", index=False)

print("\nCustomer Segmentation Results Saved Successfully!")
# Apply PCA for 2D Visualization

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("\nPCA Applied Successfully!")
print("PCA Data Shape:", X_pca.shape)
# Visualize PCA Clusters

plt.figure(figsize=(8, 5))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=df["Cluster"],
    cmap="viridis",
    s=50
)

plt.title("Customer Segmentation using PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Cluster")

plt.savefig("pca_clusters.png")
plt.close()

print("\nPCA Cluster Visualization Saved Successfully!")
# Visualize Customer Clusters

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"],
    cmap="viridis",
    s=50
)

plt.title("Customer Segmentation")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.colorbar(label="Cluster")

plt.savefig("customer_clusters.png")
plt.close()

print("\nCustomer Cluster Visualization Saved Successfully!")