from flask import Flask, request, jsonify
import joblib, os, numpy as np

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

model = joblib.load(os.path.join(MODEL_DIR, "crop_recommend_svm_soil.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "crop_scaler_soil.pkl"))
soil_encoder = joblib.load(os.path.join(MODEL_DIR, "soil_encoder.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder_soil.pkl"))

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}

@app.route("/api/ml/crop-recommendation", methods=["POST"])
def recommend_crop():
    data = request.json or {}

    soil = data.get("soil", "loamy").lower()
    rainfall = float(data.get("rainfall", 600))
    temperature = float(data.get("temperature", 25))

    soil_encoded = soil_encoder.transform([soil])[0]

    # N P K temp humidity ph rainfall soil
    X = np.array([[60, 40, 40, temperature, 70, 6.5, rainfall, soil_encoded]])
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    crop = label_encoder.inverse_transform([pred])[0]

    return jsonify({
        "crop": crop,
        "confidence": 0.75,
        "expectedYield": "Model-based estimate",
        "reasoning": "ML model prediction using soil + climate features"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
