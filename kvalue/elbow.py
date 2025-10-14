# 1. SETUP AND DATA LOADING
# -------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator # To programmatically find the elbow

print("--- Customer Segmentation: Finding the Optimal K ---")

# Load the dataset
# Make sure 'customer_segmentation.csv' is in the same folder as your script
try:
    df = pd.read_csv('../customer_segmentation_data.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'customer_segmentation.csv' not found.")
    print("Please download the dataset from: https://www.kaggle.com/datasets/fahmidachowdhury/customer-segmentation-data-for-marketing-analysis")
    exit()

# 2. DATA PREPROCESSING
# -------------------------------------
# Select features for clustering and scale them
features = ['income', 'purchase_frequency', 'last_purchase_amount']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Data has been preprocessed and scaled.\n")

# Define a range of k values to test
k_range = range(2, 11) # K must be >= 2 for silhouette and gap statistic

# --- METHOD 1: THE ELBOW METHOD ---
# -------------------------------------
print("--- 1. Running the Elbow Method ---")
wcss = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Find the elbow point programmatically
kl = KneeLocator(list(k_range), wcss, curve='convex', direction='decreasing')
elbow_k = kl.elbow

# Plot the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, marker='o', linestyle='--')
plt.axvline(x=elbow_k, color='red', linestyle='--', label=f'Optimal k = {elbow_k}')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.legend()
plt.grid(True)
plt.show()

print(f"The Elbow Method suggests the optimal k is: {elbow_k}\n")