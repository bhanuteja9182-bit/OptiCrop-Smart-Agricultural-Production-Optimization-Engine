"""
app.py
------
OptiCrop: Smart Agricultural Production Optimization Engine
Epic 5 - Application Building: Flask backend serving the web UI and the
/predict API that wraps the trained Logistic Regression model.
"""

import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

app = Flask(__name__)

model = joblib.load(os.path.join(MODELS_DIR, "crop_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

# Short farmer-facing blurb for each crop, shown alongside the prediction
CROP_INFO = {
    "rice": "Thrives in flooded fields with high humidity and warm temperatures.",
    "maize": "A versatile cereal that favours moderate rainfall and fertile loam.",
    "chickpea": "A hardy legume suited to cooler, drier growing seasons.",
    "kidneybeans": "Prefers well-drained soil and moderate rainfall.",
    "pigeonpeas": "Drought-tolerant legume that enriches soil nitrogen.",
    "mothbeans": "Extremely drought-resistant, ideal for arid regions.",
    "mungbean": "Fast-growing pulse that likes warm, humid conditions.",
    "blackgram": "Grows well in warm climates with moderate rainfall.",
    "lentil": "A cool-season pulse crop needing well-drained soil.",
    "pomegranate": "Prefers hot, dry summers and moderate winter chill.",
    "banana": "Needs consistent warmth, high humidity, and rich soil.",
    "mango": "A tropical fruit tree favouring warm, dry flowering periods.",
    "grapes": "Prefers well-drained soil, high potassium, and mild humidity.",
    "watermelon": "Needs warm temperatures and consistent moisture.",
    "muskmelon": "Thrives in warm, sunny conditions with sandy loam soil.",
    "apple": "Requires a cool climate with a distinct winter chill period.",
    "orange": "Prefers subtropical climates with well-distributed rainfall.",
    "papaya": "Fast-growing tropical fruit needing warmth and humidity.",
    "coconut": "Coastal crop that loves high humidity and sandy soil.",
    "cotton": "Needs a long, warm growing season and moderate rainfall.",
    "jute": "Prefers hot, humid climates with high rainfall.",
    "coffee": "Grows best in cool tropical highlands with shade.",
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        values = [float(data[f]) for f in FEATURES]
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Please provide valid numeric values for all fields."}), 400

    sample = pd.DataFrame([values], columns=FEATURES)
    sample_scaled = scaler.transform(sample)

    probs = model.predict_proba(sample_scaled)[0]
    top_idx = np.argsort(probs)[::-1][:3]
    top_crops = [
        {"crop": encoder.inverse_transform([i])[0], "confidence": round(float(probs[i]) * 100, 1)}
        for i in top_idx
    ]

    best_crop = top_crops[0]["crop"]
    return jsonify({
        "best_crop": best_crop,
        "confidence": top_crops[0]["confidence"],
        "info": CROP_INFO.get(best_crop, ""),
        "alternatives": top_crops[1:],
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
