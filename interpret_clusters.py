# 1. SETUP AND DATA LOADING
# -------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

print("--- Interpreting Clusters with a Decision Tree ---")

# Load the dataset with FM features
try:
    df = pd.read_csv('/home/quark/stats2/RFM/behavioral_customer_data.csv')
    print("Dataset with behavioral (FM) features loaded successfully!")
except FileNotFoundError:
    print("Error: 'behavioral_customer_data.csv' not found.")
    print("Please run the 'rfm_analysis.py' script first to generate this file.")
    exit()

# 2. FEATURE SELECTION AND SCALING
# -------------------------------------
# We will use the FM scores for clustering
fm_features = ['F_Score', 'M_Score']
X_cluster = df[fm_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)
print("FM features have been selected and scaled.\n")

# 3. LOAD OPTIMAL K AND RUN K-MEANS
# -------------------------------------
# Read the optimal_k value
k_value_file = os.path.join('kvalue', 'optimal_k.txt')
try:
    with open(k_value_file, 'r') as f:
        optimal_k = int(f.read().strip())
    print(f"Using optimal k = {optimal_k} for K-Means.\n")
except (FileNotFoundError, ValueError):
    print(f"Error: Could not read '{k_value_file}'. Please run gap_statistic.py first.")
    exit()

# Run K-Means to get the cluster labels
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to the DataFrame for profiling
df['cluster'] = cluster_labels


# 4. --- NEW: PROFILING AND LABELING CLUSTERS ---
# We do this to map cluster numbers (0, 1, 2...) to meaningful names.
print("Profiling clusters to create human-readable labels...")
segment_profiles = df.groupby('cluster')[fm_features].mean()
segment_profiles['sample_count'] = df.groupby('cluster').size()

# Calculate overall averages to use as a baseline for high/low comparison
overall_avg_f_score = df['F_Score'].mean()
overall_avg_m_score = df['M_Score'].mean()

def assign_fm_label(row):
    """Assigns a descriptive label to a cluster based on its FM profile."""
    f_level = "High" if row['F_Score'] > overall_avg_f_score else "Low"
    m_level = "High" if row['M_Score'] > overall_avg_m_score else "Low"

    # Define rules for labeling based on the feature levels
    if f_level == "High" and m_level == "High":
        return "VIP / High-Value Loyalist"
    elif f_level == "High" and m_level == "Low":
        return "Frequent Bargain Hunter"
    elif f_level == "Low" and m_level == "High":
        return "Occasional Big Spender"
    elif f_level == "Low" and m_level == "Low":
        return "At-Risk / Lapsed Customer"
    return "General" # A fallback label

# Apply the labeling function
segment_profiles['label'] = segment_profiles.apply(assign_fm_label, axis=1)

# --- MODIFIED: Create a combined list of class names ---
# This list will have names like "Cluster 0 (At-Risk)"
class_names_list = []
for cluster_num, row in segment_profiles.iterrows():
    # iterrows() on a DataFrame with an integer index will give cluster_num
    label = row['label']
    class_names_list.append(f"Cluster {cluster_num} ({label})")
    
print("Segment profiles and labels created successfully:\n", segment_profiles)


# 5. PREPARE DATA FOR DECISION TREE
# -------------------------------------
# X (Features): The FM scores (unscaled, for human-readable rules)
# y (Target): The cluster labels we just generated
X_tree = df[fm_features]
y_tree = cluster_labels

# Split data for a simple accuracy check (optional but good practice)
X_train, X_test, y_train, y_test = train_test_split(X_tree, y_tree, test_size=0.3, random_state=42)

# 6. TRAIN THE DECISION TREE
# -------------------------------------
print("\nTraining a Decision Tree to find interpretable rules...")
# We use a shallow max_depth to keep the rules simple and human-readable.
dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_classifier.fit(X_train, y_train)

# Check accuracy
accuracy = dt_classifier.score(X_test, y_test)
print(f"Decision Tree Accuracy: {accuracy*100:.2f}%")
print("Note: High accuracy means the tree's rules are a good approximation of the clustering logic.\n")
# 8. PRINT TEXT-BASED RULES
# -------------------------------------
print("\n--- Human-Readable IF-THEN Rules ---")
# Generate the text rules, e.g., "IF M_Score <= 3.5 AND F_Score <= 2.5 THEN Cluster = '...'"
rules = export_text(
    dt_classifier,
    feature_names=fm_features,
    class_names=class_names_list 
)
print(rules)
# 7. VISUALIZE THE DECISION TREE RULES
# -------------------------------------
print("Generating Decision Tree visualization...")
plt.figure(figsize=(20, 12))
plot_tree(
    dt_classifier,
    filled=True,
    rounded=True,
    feature_names=fm_features,
    class_names=class_names_list  
)
plt.title(f"Decision Tree Rules for Explaining {optimal_k} Clusters (FM Features)", fontsize=20)
plt.savefig('plots/decision_tree_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('plots/decision_tree_plot.pdf', format='pdf', bbox_inches='tight')
plt.show()

