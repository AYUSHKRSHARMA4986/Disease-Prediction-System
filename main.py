from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import json

app = FastAPI()

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load model & symptoms
model = joblib.load('ChronicDisease/symptom_model.pkl')
symptoms_list = joblib.load('ChronicDisease/symptom_list.pkl')

# ✅ Load disease info JSON
with open('ChronicDisease/disease_info.json') as f:
    disease_info = json.load(f)

# ✅ Normalize JSON keys (VERY IMPORTANT)
disease_info = {k.strip(): v for k, v in disease_info.items()}

# ✅ Sort symptoms by importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
sorted_symptoms = [symptoms_list[i] for i in indices]


# ------------------ ROUTES ------------------

@app.get("/")
def home():
    return {"message": "API is running successfully 🚀"}


@app.get("/symptoms")
def get_symptoms():
    return sorted_symptoms


@app.post("/predict")
async def predict(data: dict):
    selected_symptoms = data.get("selected_symptoms", [])

    # ✅ Create input vector
    input_vector = np.zeros(len(symptoms_list))

    for s in selected_symptoms:
        if s in symptoms_list:
            index = symptoms_list.index(s)
            input_vector[index] = 1

    df_input = pd.DataFrame([input_vector], columns=symptoms_list)

    # ✅ Get probabilities
    probs = model.predict_proba(df_input)[0]
    classes = model.classes_

    # ✅ Get top 3 predictions
    top_indices = np.argsort(probs)[::-1][:3]

    results = []

    for i in top_indices:
        disease = classes[i].strip()   # 🔥 FIX HERE
        probability = float(probs[i]) * 100

        info = disease_info.get(disease, {
            "description": "No description available",
            "precautions": []
        })

        results.append({
            "disease": disease,
            "probability": round(probability, 2),
            "description": info["description"],
            "precautions": info["precautions"]
        })

    return {
        "results": results,
        "status": "Success"
    }