# 1. SETUP AND DATA LOADING
# -------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

print("--- Applying K-Means for Customer Segmentation ---")

# --- UPDATED: Load the file that contains ALL columns, including FM scores ---
try:
    df = pd.read_csv('/home/quark/stats2/RFM/behavioral_customer_data.csv')
    print("Dataset with all features (raw + FM scores) loaded successfully!")
except FileNotFoundError:
    print("Error: 'behavioral_customer_data.csv' not found.")
    print("Please run the 'rfm_analysis.py' script first to generate this file.")
    exit()

# 2. DATA PREPROCESSING
# -------------------------------------
# Select the features for clustering (This logic is UNCHANGED)
features = ['income', 'purchase_frequency', 'last_purchase_amount']
X = df[features]

# Scale the data to ensure all features contribute equally
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Data has been preprocessed and scaled (based on income, frequency, amount).\n")


# 3. K-MEANS CLUSTERING APPLICATION
# -------------------------------------
# This logic is UNCHANGED
k_value_file = os.path.join('kvalue', 'optimal_k.txt')

try:
    with open(k_value_file, 'r') as f:
        optimal_k = int(f.read().strip())
    print(f"Successfully read optimal k = {optimal_k} from '{k_value_file}'")
except (FileNotFoundError, ValueError):
    print(f"Error: Could not read the optimal k value from '{k_value_file}'.")
    print("Please run your gap_statistic.py script first to generate this file.")
    exit()

print(f"\nApplying K-Means algorithm with k = {optimal_k}...")

# Initialize and fit the K-Means model (This logic is UNCHANGED)
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add the resulting cluster labels back to the original DataFrame
df['cluster'] = cluster_labels
print("K-Means applied and cluster labels have been assigned.\n")


# 4. PROFILING AND LABELING THE SEGMENTS
# -------------------------------------
# This is a crucial step for interpreting the results.
print("--- Customer Segment Profiles (including sample counts and labels) ---")
profile_features = ['income', 'purchase_frequency', 'last_purchase_amount', 'F_Score', 'M_Score']

# Calculate the mean for the profile features
segment_profiles = df.groupby('cluster')[profile_features].mean()
segment_profiles['sample_count'] = df.groupby('cluster').size()
overall_avg_income = df['income'].mean()
overall_avg_freq = df['purchase_frequency'].mean()
overall_avg_amount = df['last_purchase_amount'].mean()

def assign_label(row):
    """Assigns a descriptive label to a cluster based on its profile."""
    income_level = "High" if row['income'] > overall_avg_income else "Low"
    freq_level = "High" if row['purchase_frequency'] > overall_avg_freq else "Low"
    amount_level = "High" if row['last_purchase_amount'] > overall_avg_amount else "Low"

    # Define rules for labeling based on the feature levels
    if income_level == "High" and freq_level == "High":
        return "VIP / High-Value Loyalist"
    elif income_level == "High" and freq_level == "Low":
        if amount_level == "High":
            return "High-Income, Big Spender"
        else:
            return "High-Income, Low-Engagement"
    elif income_level == "Low" and freq_level == "High":
        if amount_level == "High":
            return "Frequent Spender (Mid-Value)"
        else:
            return "Frequent Bargain Hunter"
    elif income_level == "Low" and freq_level == "Low":
        if amount_level == "High":
            return "Occasional Spender"
        else:
            return "At-Risk / Lapsed Customer"
    return "General" # A fallback label

# Apply the labeling function to create the new 'label' column
segment_profiles['label'] = segment_profiles.apply(assign_label, axis=1)

# Set Pandas options to display all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(segment_profiles)

# --- ADDED: Map labels back to the main DataFrame ---
# This is needed so the final CSV has the human-readable labels
label_map = segment_profiles['label'].to_dict()
df['label'] = df['cluster'].map(label_map)
print("\nHuman-readable labels have been assigned to all customers in the DataFrame.")


# 5. VISUALIZING THE CLUSTERS
# -------------------------------------
# This logic is UNCHANGED
print("\nGenerating 3D scatter plot of the clusters...")

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Create a scatter plot for each cluster
scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], 
                     c=df['cluster'], cmap='viridis', s=50)

# Labeling the axes and title
ax.set_title(f'K-Means Clustering (k={optimal_k})')
ax.set_xlabel(f'Scaled {features[0]}')
ax.set_ylabel(f'Scaled {features[1]}')
ax.set_zlabel(f'Scaled {features[2]}')
plt.legend(*scatter.legend_elements(), title='Clusters')

# --- Save the plot as a high-quality file ---
plt.savefig('plots/kmeans_raw_features_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('plots/kmeans_raw_features_plot.pdf', format='pdf', bbox_inches='tight')
print(f"Plot saved as 'kmeans_raw_features_plot.png'")

plt.show()

output_filename = 'kmeans_segmented_customers.csv'
df.to_csv(output_filename, index=False)
print(f"\nFinal, labeled dataset (from K-Means) saved as '{output_filename}'")