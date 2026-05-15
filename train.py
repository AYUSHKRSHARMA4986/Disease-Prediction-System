import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load data
train_data = pd.read_csv('ChronicDisease/Training.csv')
test_data = pd.read_csv('ChronicDisease/Testing.csv')

# Remove unwanted column
if 'Unnamed: 133' in train_data.columns:
    train_data = train_data.drop('Unnamed: 133', axis=1)

# Split data
X_train = train_data.drop('prognosis', axis=1)
y_train = train_data['prognosis']

X_test = test_data.drop('prognosis', axis=1)
y_test = test_data['prognosis']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy * 100:.2f}%")

# Save model
joblib.dump(model, 'ChronicDisease/symptom_model.pkl')
joblib.dump(list(X_train.columns), 'ChronicDisease/symptom_list.pkl')

print("Model saved successfully!")