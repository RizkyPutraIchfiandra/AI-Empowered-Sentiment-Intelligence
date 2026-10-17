"""Flask API serving the trained sentiment model.

Run train.py first, then:  python app.py
"""

import pathlib

import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS

HERE = pathlib.Path(__file__).parent
MODEL_PATH = HERE / "model.joblib"
MAX_CHARS = 2000

app = Flask(__name__)
# Local dev only. Narrow this to your real origin before deploying.
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5173"])

if not MODEL_PATH.exists():
    raise SystemExit(f"{MODEL_PATH.name} not found. Run `python train.py` first.")

model = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return jsonify(status="ok", model=MODEL_PATH.name)


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text")

    if not isinstance(text, str) or not text.strip():
        return jsonify(error="Field `text` must be a non-empty string."), 400
    if len(text) > MAX_CHARS:
        return jsonify(error=f"Text too long. Limit is {MAX_CHARS} characters."), 413

    text = text.strip()
    proba = model.predict_proba([text])[0]
    positive = float(proba[1])
    label = "positive" if positive >= 0.5 else "negative"

    return jsonify(
        sentiment=label,
        confidence=round(max(positive, 1 - positive) * 100, 2),
        scores={"negative": round(float(proba[0]) * 100, 2), "positive": round(positive * 100, 2)},
    )


if __name__ == "__main__":
    print("Sentiment Analysis API is running on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
