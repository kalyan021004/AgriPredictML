from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ================= CROP RECOMMENDATION =================
crop_model = joblib.load(os.path.join(MODEL_DIR, "crop_recommend_svm_soil.pkl"))
crop_scaler = joblib.load(os.path.join(MODEL_DIR, "crop_scaler_soil.pkl"))
crop_label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder_soil.pkl"))
soil_encoder = joblib.load(os.path.join(MODEL_DIR, "soil_encoder.pkl"))

# ================= DISEASE DETECTION =================
disease_model = joblib.load(os.path.join(MODEL_DIR, "crop_disease_svm.pkl"))
disease_scaler = joblib.load(os.path.join(MODEL_DIR, "crop_disease_scaler.pkl"))

crop_encoder = joblib.load(os.path.join(MODEL_DIR, "crop_label_encoder.pkl"))
disease_encoder = joblib.load(os.path.join(MODEL_DIR, "disease_label_encoder.pkl"))
color_encoder = joblib.load(os.path.join(MODEL_DIR, "color_label_encoder.pkl"))
spot_color_encoder = joblib.load(os.path.join(MODEL_DIR, "spot_color_encoder.pkl"))

# ================= HEALTH =================
@app.route("/health")
def health():
    return {"status": "ok"}

# ================= CROP API =================
@app.route("/api/ml/crop-recommendation", methods=["POST"])
def crop_recommendation():
    data = request.json or {}

    N = float(data.get("N", 60))
    P = float(data.get("P", 40))
    K = float(data.get("K", 40))
    temperature = float(data.get("temperature", 25))
    humidity = float(data.get("humidity", 70))
    ph = float(data.get("ph", 6.5))
    rainfall = float(data.get("rainfall", 600))
    soil = data.get("soil", "loamy").lower()

    soil_encoded = soil_encoder.transform([soil])[0]

    X = np.array([[N, P, K, temperature, humidity, ph, rainfall, soil_encoded]])
    X_scaled = crop_scaler.transform(X)

    pred = crop_model.predict(X_scaled)[0]
    crop = crop_label_encoder.inverse_transform([pred])[0]

    return jsonify({
        "crop": crop,
        "confidence": 0.75,
        "expectedYield": "Model-based estimate",
        "reasoning": "ML prediction using soil + nutrient + climate features"
    })

# ================= DISEASE API =================
@app.route("/api/ml/disease-detection", methods=["POST"])
def disease_detection():
    data = request.json or {}

    try:
        # 1️⃣ Read inputs (DO NOT lower-case crop)
        crop = data.get("crop", "Rice").strip()
        leaf_color = data.get("leaf_color", "yellow").lower()
        spot_color = data.get("spot_color", "brown").lower()

        # 2️⃣ 🔒 VALIDATION (ADD THIS HERE)
        if crop not in crop_encoder.classes_:
            return jsonify({
                "error": "Invalid crop",
                "allowed_crops": list(crop_encoder.classes_)
            }), 400

        if leaf_color not in color_encoder.classes_:
            return jsonify({
                "error": "Invalid leaf_color",
                "allowed_leaf_colors": list(color_encoder.classes_)
            }), 400

        if spot_color not in spot_color_encoder.classes_:
            return jsonify({
                "error": "Invalid spot_color",
                "allowed_spot_colors": list(spot_color_encoder.classes_)
            }), 400

        # 3️⃣ Encode inputs
        crop_encoded = crop_encoder.transform([crop])[0]
        leaf_color_encoded = color_encoder.transform([leaf_color])[0]
        spot_color_encoded = spot_color_encoder.transform([spot_color])[0]

        # 4️⃣ Prepare features
        X = np.array([[crop_encoded, leaf_color_encoded, spot_color_encoded]])
        X_scaled = disease_scaler.transform(X)

        # 5️⃣ Predict
        pred = disease_model.predict(X_scaled)[0]
        disease = disease_encoder.inverse_transform([pred])[0]

        return jsonify({
            "disease": disease,
            "confidence": 0.78,
            "severity": "Moderate",
            "reasoning": "ML prediction using crop + leaf symptoms"
        })

    except Exception as e:
        return jsonify({
            "error": "Disease prediction failed",
            "details": str(e)
        }), 500

    data = request.json or {}

    try:
        crop = data.get("crop", "rice").lower()
        leaf_color = data.get("leaf_color", "green").lower()
        spot_color = data.get("spot_color", "none").lower()

        crop_encoded = crop_encoder.transform([crop])[0]
        leaf_color_encoded = color_encoder.transform([leaf_color])[0]
        spot_color_encoded = spot_color_encoder.transform([spot_color])[0]

        X = np.array([[crop_encoded, leaf_color_encoded, spot_color_encoded]])
        X_scaled = disease_scaler.transform(X)

        pred = disease_model.predict(X_scaled)[0]
        disease = disease_encoder.inverse_transform([pred])[0]

        return jsonify({
            "disease": disease,
            "confidence": 0.78,
            "severity": "Moderate",
            "reasoning": "ML prediction using crop + leaf symptoms"
        })

    except Exception as e:
        return jsonify({
            "error": "Disease prediction failed",
            "details": str(e)
        }), 400

# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
