# 1. SETUP AND DATA LOADING
# -------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import os

print("--- Comparing Clustering Algorithms for Customer Segmentation ---")

# Load the dataset
try:
    df = pd.read_csv('/home/quark/stats2/RFM/behavioral_customer_data.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'customer_segmentation.csv' not found.")
    exit()

# 2. DATA PREPROCESSING
# -------------------------------------
features = ['income', 'purchase_frequency', 'last_purchase_amount']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Data has been preprocessed and scaled.\n")


# 3. LOAD OPTIMAL K
# -------------------------------------
# Read the optimal_k value for K-Means and Hierarchical clustering
k_value_file = os.path.join('kvalue', 'optimal_k.txt')
try:
    with open(k_value_file, 'r') as f:
        optimal_k = int(f.read().strip())
    print(f"Using optimal k = {optimal_k} for K-Means and Hierarchical Clustering.\n")
except (FileNotFoundError, ValueError):
    print(f"Error: Could not read '{k_value_file}'. Please run gap_statistic.py first.")
    exit()


# 4. RUN ALL THREE ALGORITHMS
# -------------------------------------
# --- K-Means ---
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# --- Hierarchical Clustering ---
agg_cluster = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
hierarchical_labels = agg_cluster.fit_predict(X_scaled)

# --- DBSCAN ---
dbscan = DBSCAN(eps=0.4, min_samples=8)
dbscan_labels = dbscan.fit_predict(X_scaled)


# 5. CALCULATE COMPARISON METRICS
# -------------------------------------
# --- Silhouette Score ---
# A higher score indicates denser, more well-separated clusters.
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
hierarchical_silhouette = silhouette_score(X_scaled, hierarchical_labels)
# For DBSCAN, we only score the actual clusters, not the outliers
dbscan_core_samples_mask = dbscan_labels != -1
if np.sum(dbscan_core_samples_mask) > 1: # Score is only valid if there's more than 1 cluster
    dbscan_silhouette = silhouette_score(X_scaled[dbscan_core_samples_mask], dbscan_labels[dbscan_core_samples_mask])
else:
    dbscan_silhouette = -1 # Not applicable

# --- Cluster & Outlier Counts ---
kmeans_counts = np.bincount(kmeans_labels)
hierarchical_counts = np.bincount(hierarchical_labels)
n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_outliers_dbscan = list(dbscan_labels).count(-1)


# 6. GENERATE COMPARATIVE REPORT
# -------------------------------------
print("--- Algorithm Comparison Report ---")
report_data = {
    "Metric": ["Algorithm Type", "Number of Clusters", "Silhouette Score", "Cluster Size Distribution", "Identified Outliers"],
    "K-Means": [
        "Partitioning",
        optimal_k,
        f"{kmeans_silhouette:.4f}",
        f"Std Dev: {np.std(kmeans_counts):.2f}",
        0
    ],
    "Hierarchical": [
        "Hierarchical",
        optimal_k,
        f"{hierarchical_silhouette:.4f}",
        f"Std Dev: {np.std(hierarchical_counts):.2f}",
        0
    ],
    "DBSCAN": [
        "Density-Based",
        n_clusters_dbscan,
        f"{dbscan_silhouette:.4f}" if dbscan_silhouette != -1 else "N/A",
        "N/A",
        n_outliers_dbscan
    ]
}
report_df = pd.DataFrame(report_data)
print(report_df.to_string(index=False))


# 7. CONCLUSIVE SUMMARY
# -------------------------------------
# print("\n--- Conclusive Summary ---")
# print("Based on the analysis, the 'best' algorithm depends entirely on the business objective:\n")

# print("1. K-Means: BEST FOR GENERAL-PURPOSE MARKETING SEGMENTATION")
# print("   - Why: It produced balanced clusters with the highest Silhouette Score, indicating high-quality, well-defined groups.")
# print("   - Use Case: Creating a small number of distinct, actionable customer personas (e.g., 'VIPs', 'At-Risk') for targeted campaigns. Its results are stable and easy to interpret.\n")

# print("2. Hierarchical Clustering: BEST FOR VISUALIZING RELATIONSHIPS")
# print("   - Why: While its Silhouette Score was slightly lower, its main strength is the dendrogram, which visually explains how segments are related.")
# print("   - Use Case: Understanding the nested structure of the customer base. It provides similar personas to K-Means but with added visual context.\n")

# print("3. DBSCAN: BEST FOR ANOMALY DETECTION AND FINDING NICHE GROUPS")
# print("   - Why: This algorithm is not designed for partitioning the entire dataset. Its strength is in automatically identifying what is 'normal' and what is an 'outlier'.")
# print("   - Use Case: Identifying customers with highly unusual behavior for fraud detection, or finding small, hyper-specific niche segments that other algorithms would miss.\n")

# print("--- FINAL RECOMMENDATION ---")
# print(f"For your project's goal of creating the most 'meaningful and useful customer segments' for marketing, **K-Means is the recommended algorithm**.")
# print("It provides the most statistically sound and balanced segments, which are ideal for building a clear and actionable segmentation strategy.")


# 8. VISUAL COMPARISON OF SILHOUETTE SCORES
# -------------------------------------
print("\nGenerating visual comparison of algorithm performance...")

algorithms = ['K-Means', 'Hierarchical', 'DBSCAN']
scores = [kmeans_silhouette, hierarchical_silhouette, dbscan_silhouette]
colors = ['green', 'blue', 'orange']

plt.figure(figsize=(10, 6))
bars = plt.bar(algorithms, scores, color=colors)
plt.ylabel('Silhouette Score')
plt.title('Comparison of Clustering Algorithm Performance')
plt.ylim(min(scores) - 0.05, max(scores) + 0.05) # Adjust y-axis limits for better viewing

# Add the score value on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom' if yval > 0 else 'top')

plt.grid(axis='y', linestyle='--')
plt.savefig('comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('comparison.pdf', format='pdf', bbox_inches='tight')
plt.show()

