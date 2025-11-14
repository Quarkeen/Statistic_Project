# 1. SETUP AND DATA LOADING
# -------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import os

print("--- Applying DBSCAN for Customer Segmentation ---")

# Load the dataset
# Make sure 'customer_segmentation.csv' is in the same folder as your script
try:
    df = pd.read_csv('dataset/customer_segmentation_data.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'customer_segmentation.csv' not found.")
    print("Please download the dataset and place it in the same directory as the script.")
    exit()

# 2. DATA PREPROCESSING
# -------------------------------------
# Select the features for clustering
features = ['income', 'purchase_frequency', 'last_purchase_amount']
X = df[features]

# Scale the data to ensure all features contribute equally
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Data has been preprocessed and scaled.\n")


# 3. DBSCAN CLUSTERING APPLICATION
# -------------------------------------
# DBSCAN does not require a pre-defined k, but is sensitive to its parameters.
# 'eps' is the max distance between two samples for one to be considered as in the neighborhood of the other.
# 'min_samples' is the number of samples in a neighborhood for a point to be considered as a core point.
# --- TUNED PARAMETERS ---
# The original parameters (eps=0.5, min_samples=5) were too broad, resulting in one large cluster.
# These new, stricter parameters will help find smaller, denser, and more meaningful segments.
print("Applying DBSCAN algorithm with tuned parameters...")
dbscan = DBSCAN(eps=0.4, min_samples=8)
cluster_labels = dbscan.fit_predict(X_scaled)

# Add the resulting cluster labels back to the original DataFrame
df['cluster'] = cluster_labels
print("DBSCAN applied and cluster labels have been assigned.\n")

# --- DBSCAN specific results ---
# The label -1 is assigned to outliers (noise points)
n_clusters_ = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_outliers = list(cluster_labels).count(-1)
print(f"DBSCAN found {n_clusters_} clusters and identified {n_outliers} outliers.\n")


# 4. PROFILING AND LABELING THE SEGMENTS
# -------------------------------------
# We will profile the main clusters and also look at the characteristics of the outliers.
print("--- Customer Segment Profiles (excluding outliers) ---")

# Filter out the outliers for the main profile analysis
main_clusters_df = df[df['cluster'] != -1]

if not main_clusters_df.empty:
    segment_profiles = main_clusters_df.groupby('cluster')[features].mean()
    segment_profiles['sample_count'] = main_clusters_df.groupby('cluster').size()

    # --- Automated Labeling Logic ---
    overall_avg_income = df['income'].mean()
    overall_avg_freq = df['purchase_frequency'].mean()
    overall_avg_amount = df['last_purchase_amount'].mean()

    def assign_label(row):
        income_level = "High" if row['income'] > overall_avg_income else "Low"
        freq_level = "High" if row['purchase_frequency'] > overall_avg_freq else "Low"
        amount_level = "High" if row['last_purchase_amount'] > overall_avg_amount else "Low"
        if income_level == "High" and freq_level == "High": return "VIP / High-Value Loyalist"
        elif income_level == "High": return "High-Income Group"
        elif freq_level == "High": return "Frequent Buyer Group"
        else: return "Low-Engagement Group"

    segment_profiles['label'] = segment_profiles.apply(assign_label, axis=1)
    print(segment_profiles)
else:
    print("DBSCAN did not find any dense clusters.")

# --- Profiling the Outliers ---
outliers_df = df[df['cluster'] == -1]
if not outliers_df.empty:
    print("\n--- Profile of Outliers (Anomalies) ---")
    outlier_profile = outliers_df[features].mean().to_frame().T
    outlier_profile['sample_count'] = len(outliers_df)
    print(outlier_profile)


# 5. VISUALIZING THE CLUSTERS
# -------------------------------------
print("\nGenerating 3D scatter plot of the clusters and outliers...")

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Plot the main clusters
ax.scatter(X_scaled[df['cluster'] != -1, 0], X_scaled[df['cluster'] != -1, 1], X_scaled[df['cluster'] != -1, 2], 
           c=df[df['cluster'] != -1]['cluster'], cmap='viridis', s=50, label='Clusters')

# Plot the outliers in a distinct color (e.g., red)
ax.scatter(X_scaled[df['cluster'] == -1, 0], X_scaled[df['cluster'] == -1, 1], X_scaled[df['cluster'] == -1, 2], 
           c='red', s=25, label='Outliers')

# Labeling the axes and title
ax.set_title('DBSCAN Clustering Results')
ax.set_xlabel(f'Scaled {features[0]}')
ax.set_ylabel(f'Scaled {features[1]}')
ax.set_zlabel(f'Scaled {features[2]}')
ax.legend()
plt.show()

