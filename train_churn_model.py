# Step 2: Train and Save the AI Model 🤖

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 
import joblib # Used to save the model

# --- 1. Load the Data ---
df = pd.read_csv('telco_churn.csv')

# --- 2. Prepare the Data ---
# Select features (inputs) and target (output)
X = df[['tenure', 'MonthlyCharges', 'TotalCharges']] # Double brackets to keep X as a DataFrame
y = df['Churn'] # Single bracket for y to keep it as a Series

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Build and Train the Model ---
# We use a Random Forest Classifier, which is a powerful model for classification tasks.
# n_estimators=100 means we use 100 trees in the forest.
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model using the training data
model.fit(X_train, y_train)

# --- 4. Evaluate the Model ---
# Make predictions on the test set
y_pred = model.predict(X_test)
accuracy_score = accuracy_score(y_test, y_pred)  # Calculate the accuracy of the model, the best score is 1.0 (100% accuracy)

print("\n---Model Training and Evaluation Complete---")
print(f"Model accuracy: {accuracy_score:.4f}")  # Display the accuracy score, formatted to 4 decimal places.

# --- 5. Save the Model ---
# Save the trained model to a file using joblib
# We save the trained model to a file so our dashboard can use it later for predictions.
joblib.dump(model, 'churn_model.pkl')

print("\n---Model Saved Successfully---")
# Now we can use this model in our dashboard to predict whether a customer will churn based on their tenure, monthly charges, and total charges.
