import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set seed and create dataset
np.random.seed(42)
rows = 200
age = np.random.randint(18, 61, rows)
monthly_income = np.random.uniform(2, 30, rows).round(2)
website_visits = np.random.randint(1, 21, rows)
score = -7 + 0.05 * age + 0.18 * monthly_income + 0.25 * website_visits
probability = 1 / (1 + np.exp(-score))
purchased = np.random.binomial(1, probability)

df = pd.DataFrame({
    'age': age,
    'monthly_income_lakh': monthly_income,
    'website_visits': website_visits,
    'purchased': purchased
})

# Split data
feature_columns = ['age', 'monthly_income_lakh', 'website_visits']
X = df[feature_columns]
y = df['purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Not Purchased', 'Purchased'], output_dict=True)

# Exercise customers
exercise_customers = pd.DataFrame({
    'age': [28, 48],
    'monthly_income_lakh': [8, 24],
    'website_visits': [5, 16]
})
predictions = model.predict(exercise_customers)
probabilities = model.predict_proba(exercise_customers)[:, 1] * 100

print('Accuracy:', accuracy)
print('Confusion Matrix:', cm.tolist())
print('Precision for Purchased:', report['Purchased']['precision'])
print('Recall for Purchased:', report['Purchased']['recall'])
print('New Customer Predicted Class:', predictions.tolist())
print('New Customer Purchase Probability:', probabilities.round(2).tolist())