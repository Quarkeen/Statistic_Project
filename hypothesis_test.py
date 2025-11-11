# 1. SETUP AND IMPORTS
# -------------------------------------
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats
import os
import numpy as np
import joblib  # To load our ML model

print("--- Statistical Hypothesis Testing Model (ML-Powered) ---")

# 2. DATA AND MODEL LOADING
# -------------------------------------
try:
    df = pd.read_csv('/home/quark/stats2/RFM/behavioral_customer_data.csv')
    print("Dataset with behavioral (FM) features loaded successfully!")
    
    # Load the ML model and encoder
    ml_model = joblib.load('test_selector_model.joblib')
    encoder = joblib.load('test_selector_encoder.joblib')
    print("ML model ('test_selector_model.joblib') and encoder loaded.")
    
    # Load the optimal k value
    k_value_file = os.path.join('kvalue', 'optimal_k.txt')
    with open(k_value_file, 'r') as f:
        optimal_k = int(f.read().strip())
    print(f"Using optimal k = {optimal_k} from 'kvalue/optimal_k.txt'")
    
except FileNotFoundError as e:
    print(f"Error: A required file was not found.")
    print(f"Details: {e}")
    print("Please make sure you have run 'rfm_analysis.py', 'gap_statistic.py', and 'train_test_selector.py' first.")
    exit()

# Re-run K-Means to get the cluster labels
fm_features = ['F_Score', 'M_Score']
X_cluster = df[fm_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)
print(f"Successfully re-ran K-Means and assigned {optimal_k} cluster labels.\n")

# Define data types for our "expert system"
DATA_TYPES = {
    'income': 'continuous',
    'purchase_frequency': 'continuous',
    'last_purchase_amount': 'continuous',
    'F_Score': 'ordinal',
    'M_Score': 'ordinal',
    'gender': 'categorical',
    'cluster': 'categorical'
}
ALPHA = 0.05

# 3. STATISTICAL HELPER FUNCTIONS
# -------------------------------------
def run_manual_mannwhitneyu(data1, data2):
    """
    Implements the Mann-Whitney U test logic from your guide,
    and calculates a p-value using the Z-test approximation.
    """
    # Ensure inputs are lists/iterables
    n1 = int(len(data1))
    n2 = int(len(data2))

    # Combine data and keep track of group membership
    combined_values = list(data1) + list(data2)
    groups = [1] * n1 + [2] * n2
    df_comb = pd.DataFrame({
        'value': combined_values,
        'group': groups
    })

    # Rank the combined values in ascending order.
    # method='average' ensures ties receive the average of the ranks they occupy.
    df_comb['rank'] = df_comb['value'].rank(method='average', ascending=True)

    # Sum ranks for group 1
    r1 = df_comb.loc[df_comb['group'] == 1, 'rank'].sum()

    # U statistic for group 1
    u1 = (n1 * n2) + (n1 * (n1 + 1) / 2.0) - r1

    # Large-sample approximation (Z-test) with tie correction for variance
    N = n1 + n2
    mu_u = (n1 * n2) / 2.0

    # Tie correction: sum over tie groups t_i of (t_i^3 - t_i)
    ties = df_comb['value'].value_counts()
    sum_ties = 0
    for t in ties:
        if t > 1:
            sum_ties += (t**3 - t)

    # Variance with tie correction
    # Var(U) = n1*n2/12 * (N+1 - sum_ties/(N*(N-1)))
    if N > 1:
        correction = sum_ties / (N * (N - 1)) if (N * (N - 1)) != 0 else 0
        var_u = (n1 * n2 / 12.0) * (N + 1 - correction)
    else:
        var_u = 0.0

    sigma_u = np.sqrt(var_u) if var_u > 0 else 0.0

    # Continuity correction (0.5) applied; handle sigma == 0
    if sigma_u == 0:
        # Degenerate case (no variance) -- fallback to non-significant
        p_value = 1.0
    else:
        z = (u1 - mu_u - 0.5) / sigma_u
        p_value = 2 * stats.norm.sf(abs(z))

    return u1, p_value
def interpret_p_value(p_value, test_name):
    """Provides a plain-English interpretation of the p-value."""
    print(f"\nTest Performed: {test_name} (as predicted by ML model)")
    if p_value < ALPHA:
        return (f"P-value is {p_value:.4f} (which is < {ALPHA}).\n"
                "Conclusion: **We REJECT the Null Hypothesis.**\n"
                "Meaning: The difference is statistically significant. Your hypothesis is ACCEPTED.")
    else:
        return (f"P-value is {p_value:.4f} (which is >= {ALPHA}).\n"
                "Conclusion: **We FAIL TO REJECT the Null Hypothesis.**\n"
                "Meaning: The difference is NOT statistically significant. Your hypothesis is REJECTED.")

# 4. THE "MODEL": A STATISTICAL DECISION MAKER
# -------------------------------------
def run_hypothesis_test(hypothesis):
    """
    This function uses the ML model to select the correct test,
    then executes it.
    """
    print("="*50)
    print(f"Running Hypothesis Test: \"{hypothesis['name']}\"")
    print("="*50)
    
    test_type = hypothesis['test_type']
    variable = hypothesis['variable']
    variable_type = DATA_TYPES.get(variable)

    # --- 1. PREPARE FEATURES FOR THE ML MODEL ---
    if test_type == 'check_association':
        data_type_feature = 'categorical'
        num_groups_feature = '2' # Chi-Squared handles 2 or many
    else:
        data_type_feature = variable_type
        if test_type == 'compare_two_groups':
            num_groups_feature = '2'
        elif test_type == 'compare_many_groups':
            num_groups_feature = 'many'
            
    # Create the feature vector for our model
    features_df = pd.DataFrame([[data_type_feature, num_groups_feature]], 
                               columns=['data_type', 'num_groups'])
    features_encoded = encoder.transform(features_df)
    
    # --- 2. GET PREDICTION FROM THE ML MODEL ---
    predicted_test = ml_model.predict(features_encoded)[0]
    print(f"ML Model Input: (data_type='{data_type_feature}', num_groups='{num_groups_feature}')")
    print(f"ML Model Prediction: Run the '{predicted_test}' test.")

    # --- 3. EXECUTE THE PREDICTED TEST ---
    if predicted_test == 'Mann-Whitney U':
        group1_name = hypothesis['group1']
        group2_name = hypothesis['group2']
        data1 = df[df['cluster'] == group1_name][variable]
        data2 = df[df['cluster'] == group2_name][variable]
        
        print(f"Comparing '{variable}' for Cluster {group1_name} (n={len(data1)}) vs. Cluster {group2_name} (n={len(data2)})")
        stat, p_value = run_manual_mannwhitneyu(data1,data2)
        print(interpret_p_value(p_value, "Mann-Whitney U Test"))

    elif predicted_test == 'Kruskal-Wallis':
        groups = df['cluster'].unique()
        group_data = [df[df['cluster'] == g][variable] for g in groups]
        
        print(f"Comparing '{variable}' across all {len(groups)} clusters.")
        stat, p_value = stats.kruskal(*group_data)
        print(interpret_p_value(p_value, "Kruskal-Wallis Test"))

    elif predicted_test == 'Chi-Squared':
        var2 = hypothesis['variable2']
        print(f"Checking association between '{variable}' and '{var2}'.")
        contingency_table = pd.crosstab(df[variable], df[var2])
        
        print("\nContingency Table:\n", contingency_table)
        stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        print(interpret_p_value(p_value, "Chi-Squared Test of Independence"))
    
    print("="*50 + "\n")


# 5. HOW TO USE THE MODEL
# -------------------------------------
# Define your hypotheses as Python dictionaries.
# Then, pass them to the run_hypothesis_test() function.

# !! IMPORTANT: Adjust these cluster numbers based on your profile labels !!
# From your K-Means output, find the numbers for your segments.
# Example: Let's assume Cluster 0 is "VIP" and Cluster 3 is "At-Risk"
VIP_CLUSTER_NUM = 0 
HIGH_INCOME_LOW_ENGAGEMENT= 3

hypothesis1 = {
    "name": "Income: VIPs vs. Low-Engagement",
    "test_type": "compare_two_groups",  # 2 groups
    "variable": "income",               # 'continuous' data
    "group1": VIP_CLUSTER_NUM,
    "group2": HIGH_INCOME_LOW_ENGAGEMENT
}
# ML Model should predict 'Mann-Whitney U'
run_hypothesis_test(hypothesis1)


# --- EXAMPLE 2: Comparing 'F_Score' across ALL clusters ---
hypothesis2 = {
    "name": "Frequency Score Across All Segments",
    "test_type": "compare_many_groups", # 'many' groups
    "variable": "F_Score"               # 'ordinal' data
}
# ML Model should predict 'Kruskal-Wallis'
# run_hypothesis_test(hypothesis2)


# --- EXAMPLE 3: Checking association between 'gender' and 'cluster' ---
hypothesis3 = {
    "name": "Gender Distribution Across Segments",
    "test_type": "check_association",   # Triggers 'categorical'
    "variable": "gender",               # 'categorical' data
    "variable2": "cluster"
}
# ML Model should predict 'Chi-Squared'
# run_hypothesis_test(hypothesis3)

# --- HOW TO RUN ---
print("--- How to Use This Script ---")
print("1. Run 'train_test_selector.py' ONCE to train the ML model.")
print("2. Look at your K-Means profiling output to match labels (e.g., 'VIP') to cluster numbers (e.g., 0).")
print("3. Update the cluster numbers in the example dictionaries (e.g., VIP_CLUSTER_NUM).")
print("4. Uncomment one of the 'run_hypothesis_test(...)' lines at the end of this script to run a test.")