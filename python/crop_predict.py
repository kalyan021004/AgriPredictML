import os
import sys
import json
import numpy as np
import joblib
import warnings

# Silence sklearn warnings (important)
warnings.filterwarnings("ignore")

try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(BASE_DIR, "models")

    model = joblib.load(os.path.join(MODEL_DIR, "crop_recommend_svm_soil.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "crop_scaler_soil.pkl"))
    soil_encoder = joblib.load(os.path.join(MODEL_DIR, "soil_encoder.pkl"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder_soil.pkl"))

    data = json.loads(sys.argv[1])

    # SAFE DEFAULTS
    nitrogen = float(data.get("nitrogen", 60))
    phosphorus = float(data.get("phosphorus", 40))
    potassium = float(data.get("potassium", 40))
    temperature = float(data.get("temperature", 25))
    humidity = float(data.get("humidity", 70))
    ph = float(data.get("ph", 6.5))
    rainfall = float(data.get("rainfall", 600))
    soil = data.get("soil", "loamy").lower()

    soil_encoded = soil_encoder.transform([soil])[0]

    X = np.array([[ 
        nitrogen,
        phosphorus,
        potassium,
        temperature,
        humidity,
        ph,
        rainfall,
        soil_encoded
    ]])

    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    prob = float(np.max(model.predict_proba(X_scaled)))
    crop = label_encoder.inverse_transform([pred])[0]

    print(json.dumps({
        "crop": crop,
        "confidence": round(prob, 2),
        "expectedYield": "Model-based estimate",
        "reasoning": "Prediction generated using SVM with soil feature"
    }))

except Exception as e:
    # 🔥 CRITICAL: NEVER CRASH, ALWAYS RETURN JSON
    print(json.dumps({
        "error": "ML prediction failed",
        "details": str(e)
    }))
