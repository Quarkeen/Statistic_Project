# 1. SETUP AND DATA LOADING
# -------------------------------------
import pandas as pd
import os
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

print("--- Behavioral Feature Engineering (Frequency-Monetary) Script ---")

# Load the original dataset
try:
    df = pd.read_csv('../customer_segmentation_data.csv')
    print("Original dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'customer_segmentation.csv' not found.")
    print("Please ensure the original dataset is in the same directory.")
    exit()

# 2. FM (FREQUENCY-MONETARY) CALCULATION
# -------------------------------------
print("\nPerforming FM analysis to create behavior-based features...")
print("Note: The dataset lacks a date column, so a true 'Recency' score cannot be calculated.")

# --- FIX: Use only available behavioral features for scoring ---
# Frequency: Higher 'purchase_frequency' is better.
# Monetary: Higher 'last_purchase_amount' is better.
df['Frequency'] = df['purchase_frequency']
df['Monetary'] = df['last_purchase_amount']

# Create FM quantiles/scores (from 1 to 5)
# Note: pd.qcut creates equal-sized bins, so a flat distribution in the plot below is expected.
df['F_Score'] = pd.qcut(df['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
df['M_Score'] = pd.qcut(df['Monetary'], 5, labels=[1, 2, 3, 4, 5])

# Convert scores to numeric type
df['F_Score'] = df['F_Score'].astype(int)
df['M_Score'] = df['M_Score'].astype(int)

print("FM scores calculated successfully.")

# 3. SAVE THE NEW DATASET
# -------------------------------------
output_filename = 'behavioral_customer_data.csv'
df.to_csv(output_filename, index=False)
print(f"\nAnalysis complete. The new dataset with FM scores has been saved as '{output_filename}'")


# 4. VISUALIZE THE FM SEGMENTS
# -------------------------------------
print("\nGenerating visualizations to interpret the FM segments...")
sns.set(style="whitegrid")



# --- Plot 2: FM Segmentation Heatmap (More Intuitive) ---
# This heatmap is a much clearer way to see the concentration of customers.
fm_grid = df.groupby(['F_Score', 'M_Score']).size().unstack()

plt.figure(figsize=(12, 8))
sns.heatmap(fm_grid, annot=True, fmt='d', cmap='YlGnBu', linewidths=.5)

plt.title('FM Segmentation Heatmap', fontsize=16)
plt.xlabel('Monetary Score (Higher is better)', fontsize=12)
plt.ylabel('Frequency Score (Higher is better)', fontsize=12)
plt.gca().invert_yaxis() # Place high frequency (5) at the top
plt.show()

print("\n--- How to Interpret the Heatmap ---")
print("The brighter the color in a square, the more customers fall into that segment.")
print("The numbers in each square show the exact count of customers.")
print("Your most valuable customers are in the top-right corner (High F, High M).")

