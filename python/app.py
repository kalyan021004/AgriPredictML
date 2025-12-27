from flask import Flask, request, jsonify
import joblib, os, numpy as np

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")

model = joblib.load(os.path.join(MODEL_DIR, "crop_recommend_svm_soil.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "crop_scaler_soil.pkl"))
soil_encoder = joblib.load(os.path.join(MODEL_DIR, "soil_encoder.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder_soil.pkl"))

@app.route("/api/ml/crop-recommendation", methods=["POST"])
def recommend_crop():
    data = request.json or {}

    try:
        # 🔹 REQUIRED INPUTS
        N = float(data["nitrogen"])
        P = float(data["phosphorus"])
        K = float(data["potassium"])
        temperature = float(data["temperature"])
        humidity = float(data["humidity"])
        ph = float(data["ph"])
        rainfall = float(data["rainfall"])
        soil = data["soil"].lower()

        soil_encoded = soil_encoder.transform([soil])[0]

        # 🔹 FEATURE VECTOR (MATCHES TRAINING ORDER)
        X = np.array([[N, P, K, temperature, humidity, ph, rainfall, soil_encoded]])
        X_scaled = scaler.transform(X)

        pred = model.predict(X_scaled)[0]
        crop = label_encoder.inverse_transform([pred])[0]

        return jsonify({
            "crop": crop,
            "confidence": round(float(np.max(model.decision_function(X_scaled))), 2),
            "expectedYield": "Model-based estimate",
            "reasoning": "Prediction generated using trained SVM with full feature set"
        })

    except KeyError as e:
        return jsonify({
            "error": f"Missing input field: {str(e)}"
        }), 400

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7001)
