"""Flask API serving the trained sentiment models.

Picks a model per request from the detected language, so an Indonesian
sentence is never scored by the English model.

Run train.py first, then:  python app.py
"""

import pathlib

import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS

from lang import detect

HERE = pathlib.Path(__file__).parent
MAX_CHARS = 2000

LANGUAGES = {
    "en": {"name": "English", "path": HERE / "model_en.joblib"},
    "id": {"name": "Indonesian", "path": HERE / "model_id.joblib"},
}

app = Flask(__name__)
# Local dev only. Narrow this to your real origin before deploying.
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5173"])

missing = [str(cfg["path"].name) for cfg in LANGUAGES.values() if not cfg["path"].exists()]
if missing:
    raise SystemExit(f"Missing {', '.join(missing)}. Run `python train.py` first.")

models = {code: joblib.load(cfg["path"]) for code, cfg in LANGUAGES.items()}


@app.get("/health")
def health():
    return jsonify(status="ok", languages={code: cfg["name"] for code, cfg in LANGUAGES.items()})


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text")

    if not isinstance(text, str) or not text.strip():
        return jsonify(error="Field `text` must be a non-empty string."), 400
    if len(text) > MAX_CHARS:
        return jsonify(error=f"Text too long. Limit is {MAX_CHARS} characters."), 413

    text = text.strip()

    # Allow an explicit override, but validate it rather than trusting it.
    requested = payload.get("language")
    if requested is not None and requested not in LANGUAGES:
        return jsonify(error=f"Unknown language `{requested}`. Use one of: {', '.join(LANGUAGES)}."), 400

    language = requested or detect(text)

    if language not in LANGUAGES:
        # Returning a coin-flip prediction on text in an untrained language
        # looks like an answer but carries no information, so say so instead.
        return (
            jsonify(
                error="Could not tell which language this is. Supported: English, Indonesian.",
                detected="unknown",
            ),
            422,
        )

    proba = models[language].predict_proba([text])[0]
    positive = float(proba[1])

    return jsonify(
        sentiment="positive" if positive >= 0.5 else "negative",
        confidence=round(max(positive, 1 - positive) * 100, 2),
        scores={"negative": round(float(proba[0]) * 100, 2), "positive": round(positive * 100, 2)},
        language=language,
        languageName=LANGUAGES[language]["name"],
        detected=requested is None,
    )


if __name__ == "__main__":
    loaded = ", ".join(f"{code} ({cfg['name']})" for code, cfg in LANGUAGES.items())
    print("Sentiment Analysis API is running on http://127.0.0.1:5000")
    print(f"Models loaded: {loaded}")
    app.run(host="127.0.0.1", port=5000, debug=False)
