import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os # Import the os module to handle directories and files

print("--- Running Gap Statistic for Optimal K ---")

# 1. DATA LOADING AND PREPROCESSING
# -------------------------------------
try:
    # Since this script is inside the 'kvalue' directory, we look for the CSV
    # in the parent directory using '../'
    df = pd.read_csv('../customer_segmentation_data.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'customer_segmentation.csv' not found in the parent directory.")
    print("Please ensure your directory structure is: parent_folder/customer_segmentation.csv and parent_folder/kvalue/gap_statistic.py")
    exit()

# Select the features for clustering
features = ['income', 'purchase_frequency', 'last_purchase_amount']
X = df[features]

# Scale the data - this is crucial for distance-based clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Data has been preprocessed and scaled.\n")


# 2. GAP STATISTIC CALCULATION
# -------------------------------------
def calculate_gap_statistic(data, k_range=range(2, 11), n_refs=20):
    """
    Calculates the Gap Statistic for a range of k values.
    Compares the WCSS of the actual data to the WCSS of randomly generated data.
    """
    print("Calculating Gap Statistic... (this may take a moment)")
    gaps = []
    s_k = []
    
    for k in k_range:
        # --- Step 1: Calculate WCSS for the actual data ---
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(data)
        wcss_actual = np.log(kmeans.inertia_)
        
        # --- Step 2: Calculate WCSS for multiple reference datasets ---
        wcss_refs = []
        for _ in range(n_refs):
            # Generate a random "null" dataset with no clustering structure
            random_data = np.random.rand(*data.shape)
            mins, maxs = data.min(axis=0), data.max(axis=0)
            random_data = mins + (maxs - mins) * random_data
            
            # Fit KMeans to the random data
            kmeans_ref = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
            kmeans_ref.fit(random_data)
            wcss_refs.append(np.log(kmeans_ref.inertia_))
        
        # --- Step 3: Calculate the Gap and the standard deviation ---
        expected_wcss = np.mean(wcss_refs)
        std_dev = np.std(wcss_refs)
        
        # The Gap is the difference between the expected (random) WCSS and the actual WCSS
        gaps.append(expected_wcss - wcss_actual)
        
        # Calculate the standard error for the error bars in the plot
        s_k.append(std_dev * np.sqrt(1 + 1/n_refs))
        
    return gaps, s_k, k_range

# Run the calculation
gaps, s_k_values, k_range = calculate_gap_statistic(X_scaled)

# Find the optimal k by locating the k with the maximum gap value.
# This approach finds the point of maximum separation from the null distribution.
gap_k = k_range[np.argmax(gaps)]

output_file = 'optimal_k.txt'

# Write the optimal k value to the specified file
try:
    with open(output_file, 'w') as f:
        f.write(str(gap_k))
    print(f"Successfully saved optimal k = {gap_k} to 'kvalue/{output_file}'")
except IOError as e:
    print(f"Error: Could not write to file '{output_file}'. Reason: {e}")

print(f"The Gap Statistic method suggests the optimal k is: {gap_k}")
# 3. PLOTTING THE RESULTS
# -------------------------------------
plt.figure(figsize=(10, 6))
# Plot the gap values with error bars
plt.errorbar(k_range, gaps, yerr=s_k_values, marker='o', linestyle='--', capsize=5, color='b')
# Highlight the optimal k value
plt.axvline(x=gap_k, color='red', linestyle='--', label=f'Optimal k = {gap_k}')
plt.title('Gap Statistic for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Gap Value')
plt.legend()
plt.grid(True)
plt.savefig('gap_statistic_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('gap_statistic_plot.pdf', format='pdf', bbox_inches='tight')
plt.show()

# 4. FINAL OUTPUT
# -------------------------------------



# 5. EXPORTING THE OPTIMAL K VALUE
# -------------------------------------
# Since this script is already inside the 'kvalue' directory,
# we can write the output file directly to the current location.



