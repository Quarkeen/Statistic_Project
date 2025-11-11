import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import joblib

print("--- Training ML Model to Select Statistical Tests ---")

# 1. Create the training data based on your stats guide's rules
# We are creating a "synthetic" dataset of scenarios
scenarios = {
    'data_type': [
        'categorical', 'categorical',  # Chi-Squared scenarios
        'ordinal', 'ordinal',          # Mann-Whitney U / Kruskal-Wallis scenarios
        'continuous', 'continuous'     # Mann-Whitney U / Kruskal-Wallis scenarios
    ],
    'num_groups': [
        '2', 'many',  # Chi-Squared works for 2 or many groups
        '2', 'many',
        '2', 'many'
    ]
}
df_train = pd.DataFrame(scenarios)

# 2. Create the labels (the correct test for each scenario)
def get_correct_test(row):
    if row['data_type'] == 'categorical':
        return 'Chi-Squared'
    if row['data_type'] in ['ordinal', 'continuous'] and row['num_groups'] == '2':
        return 'Mann-Whitney U'
    if row['data_type'] in ['ordinal', 'continuous'] and row['num_groups'] == 'many':
        return 'Kruskal-Wallis'
    return 'Unknown'

df_train['correct_test'] = df_train.apply(get_correct_test, axis=1)

print("Generated training data based on your stats guide:")
print(df_train)
print("\n")

# 3. Prepare data for the Machine Learning model
# ML models need numbers, not text. We use One-Hot Encoding.
features = df_train[['data_type', 'num_groups']]
labels = df_train['correct_test']

# Create and fit the encoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
features_encoded = encoder.fit_transform(features)

# 4. Train the Decision Tree (the "simple ML algorithm")
# We are training it to learn the rules from the data we just made
ml_model = DecisionTreeClassifier(random_state=42)
ml_model.fit(features_encoded, labels)

# 5. Check Accuracy (should be 100% because it's a simple, logical task)
predictions = ml_model.predict(features_encoded)
accuracy = accuracy_score(labels, predictions)
print(f"ML Model Accuracy on training data: {accuracy*100:.0f}%")
print("An accuracy of 100% is expected, as it is learning a simple, deterministic set of rules.")

# 6. Save the trained model and the encoder
joblib.dump(ml_model, 'test_selector_model.joblib')
joblib.dump(encoder, 'test_selector_encoder.joblib')

print("\nSuccessfully trained and saved the following files:")
print("  - test_selector_model.joblib (The trained ML model)")
print("  - test_selector_encoder.joblib (The encoder for the model)")
print("\nYou can now run 'hypothesis_tester.py' to use this model.")