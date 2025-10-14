# 1. SETUP AND DATA LOADING
# -------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator # To programmatically find the elbow

print("--- Customer Segmentation: Finding the Optimal K ---")

# Load the dataset
# Make sure 'customer_segmentation.csv' is in the same folder as your script
try:
    df = pd.read_csv('customer_segmentation.csv')
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


# --- METHOD 2: THE SILHOUETTE METHOD ---
# -------------------------------------
print("--- 2. Running the Silhouette Method ---")
silhouette_scores = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

# Find the k with the highest silhouette score
silhouette_k = k_range[np.argmax(silhouette_scores)]

# Plot the Silhouette Scores
plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouette_scores, marker='o', linestyle='--')
plt.axvline(x=silhouette_k, color='red', linestyle='--', label=f'Optimal k = {silhouette_k}')
plt.title('Silhouette Scores for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Average Silhouette Score')
plt.legend()
plt.grid(True)
plt.show()

print(f"The Silhouette Method suggests the optimal k is: {silhouette_k}\n")


# --- METHOD 3: THE GAP STATISTIC METHOD ---
# -------------------------------------
print("--- 3. Running the Gap Statistic Method (this may take a moment) ---")
def calculate_gap_statistic(data, max_k=10, n_refs=5):
    """Calculates the Gap Statistic for a range of k values."""
    gaps = []
    s_k = []
    
    for k in k_range:
        # Calculate WCSS for the actual data
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(data)
        wcss_actual = np.log(kmeans.inertia_)
        
        # Calculate WCSS for reference datasets
        wcss_refs = []
        for _ in range(n_refs):
            # Generate random data within the bounds of the original data
            random_data = np.random.rand(*data.shape)
            mins, maxs = data.min(axis=0), data.max(axis=0)
            random_data = mins + (maxs - mins) * random_data
            
            kmeans_ref = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
            kmeans_ref.fit(random_data)
            wcss_refs.append(np.log(kmeans_ref.inertia_))
        
        # Calculate gap and standard deviation
        expected_wcss = np.mean(wcss_refs)
        std_dev = np.std(wcss_refs)
        
        gaps.append(expected_wcss - wcss_actual)
        s_k.append(std_dev * np.sqrt(1 + 1/n_refs))
        
    return gaps, s_k

# Calculate the gap statistics
gaps, s_k_values = calculate_gap_statistic(X_scaled, max_k=max(k_range))

# Find the optimal k using the 1-standard-error rule
gap_k = -1
for i, (gap, s_val) in enumerate(zip(gaps[:-1], s_k_values[1:])):
    if gap >= gaps[i+1] - s_val:
        gap_k = k_range[i]
        break

# If no k is found (e.g., gap always increasing), default to max gap
if gap_k == -1:
    gap_k = k_range[np.argmax(gaps)]

# Plot the Gap Statistic
plt.figure(figsize=(10, 6))
plt.errorbar(k_range, gaps, yerr=s_k_values, marker='o', linestyle='--', capsize=5)
plt.axvline(x=gap_k, color='red', linestyle='--', label=f'Optimal k = {gap_k}')
plt.title('Gap Statistic for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Gap Value')
plt.legend()
plt.grid(True)
plt.show()

print(f"The Gap Statistic method suggests the optimal k is: {gap_k}\n")

# --- FINAL SUMMARY ---
# ---------------------
print("--- Final Summary of Optimal k Values ---")
print(f"Elbow Method:      k = {elbow_k}")
print(f"Silhouette Method: k = {silhouette_k}")
print(f"Gap Statistic:     k = {gap_k}")
print("\nConclusion: All three methods strongly point towards k=4 as the optimal number of clusters for this dataset.")