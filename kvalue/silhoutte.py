import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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
plt.savefig('../plots/silhouette_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('../plotssilhouette_plot.pdf', format='pdf', bbox_inches='tight')
plt.show()

print(f"The Silhouette Method suggests the optimal k is: {silhouette_k}\n")