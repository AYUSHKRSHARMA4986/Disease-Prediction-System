import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load('ChronicDisease/symptom_model.pkl')
symptoms = joblib.load('ChronicDisease/symptom_list.pkl')

# Create fast lookup
symptom_index = {s: i for i, s in enumerate(symptoms)}

def predict_disease(user_symptoms):
    input_vector = np.zeros(len(symptoms))

    for s in user_symptoms:
        if s in symptom_index:
            input_vector[symptom_index[s]] = 1
        else:
            print(f"Warning: {s} not found")

    df_input = pd.DataFrame([input_vector], columns=symptoms)

    probs = model.predict_proba(df_input)[0]
    classes = model.classes_

    top_indices = np.argsort(probs)[::-1][:3]

    print("\nTop Predictions:")
    for i in top_indices:
        print(f"{classes[i]} → {probs[i]*100:.2f}%")

# Test
predict_disease(['chills', 'vomiting', 'high_fever'])