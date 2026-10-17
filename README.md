<div align="center">

# Moodreader

**Read the mood before you reply.**

Paste a review or message in English or Indonesian. The language is detected, then a TF-IDF and logistic regression model scores it as positive or negative and shows how certain it is.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1-000000?logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?logo=scikitlearn&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-8-646CFF?logo=vite&logoColor=white)
![Tailwind](https://img.shields.io/badge/Tailwind-4-06B6D4?logo=tailwindcss&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-3B82F6)

<img src="docs/assets/screenshot.png" alt="Moodreader reading a product review as happy, with 67.7% confidence" width="100%" />

</div>

---

## What it does

- **Classifies English and Indonesian text** as positive or negative with a confidence percentage.
- **Detects the language per request**, so an Indonesian sentence is never scored by the English model. An explicit `language` override is accepted and validated.
- **Shows both class scores**, so a 51/49 split is visibly different from a 95/5 one.
- **Handles negation.** "Not good at all" lands on negative, because the model keeps words like *not*, *no*, *tidak*, and *banget* instead of stripping them as stop words.
- **Fails loudly, not silently.** Empty input, oversized input, text in an untrained language, and an unreachable API each get their own message.

## Built with

| Technology | What it does here |
| --- | --- |
| [scikit-learn](https://scikit-learn.org) | TF-IDF features into logistic regression, wrapped in one `Pipeline`. |
| [Flask](https://flask.palletsprojects.com) + [Flask-CORS](https://flask-cors.readthedocs.io) | Serves `/predict` and `/health` to the browser on localhost. |
| [joblib](https://joblib.readthedocs.io) | Persists each fitted pipeline as `model_en.joblib` and `model_id.joblib`. |
| [React](https://react.dev) 19 + [Vite](https://vite.dev) 8 | Interface and dev server with fast rebuilds. |
| [Tailwind CSS](https://tailwindcss.com) 4 | CSS-first theming through `@theme`, no config file. |

## Getting started

### Prerequisites

- Python 3.10 or newer
- Node.js 20 or newer

### 1. Train the models

```bash
cd backend
pip install -r requirements.txt
python train.py
```

`train.py` downloads both datasets into `backend/data/`, trains one pipeline per language, prints a classification report for each, and writes `model_en.joblib` and `model_id.joblib`. Each model ends with a self-check that reloads the saved file and asserts two obvious cases, so a broken artifact fails here rather than at request time.

| Language | Dataset | Labelled texts |
| --- | --- | --- |
| English | [UCI Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences) — Amazon, IMDb, Yelp | 3,000 |
| Indonesian | [IndoNLU SmSA](https://github.com/IndoNLP/indonlu) — Prosa document sentiment | 10,981 |

SmSA ships a third `neutral` label. It is dropped so both models stay binary and comparable; a neutral class needs its own UI treatment rather than a silent extra label.

Expected output:

```
en
3000 labelled texts  |  positive=1500  negative=1500
Accuracy: 0.8467
saved model_en.joblib
self-check passed

id
10981 labelled texts  |  positive=7151  negative=3830
Accuracy: 0.9135
saved model_id.joblib
self-check passed
```

### 2. Run the API

```bash
python app.py
```

Serves on `http://127.0.0.1:5000`. Check it with:

```bash
curl http://127.0.0.1:5000/health
```

### 3. Run the interface

In a second terminal, leaving the API running:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## API

### `GET /health`

```json
{ "status": "ok", "languages": { "en": "English", "id": "Indonesian" } }
```

### `POST /predict`

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The build quality is outstanding and it arrived two days early."}'
```

```json
{
  "sentiment": "positive",
  "confidence": 67.7,
  "scores": { "positive": 67.7, "negative": 32.3 },
  "language": "en",
  "languageName": "English",
  "detected": true
}
```

`detected` is `true` when the language came from detection, `false` when you supplied it. To skip detection, pass `language`:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Pelayanan nya cepat dan ramah.", "language": "id"}'
```

| Status | When |
| --- | --- |
| `200` | Prediction returned. |
| `400` | `text` missing, not a string, or blank; or `language` is not `en` or `id`. |
| `413` | `text` longer than 2,000 characters. |
| `422` | Language could not be determined and none was supplied. |

## How the model works

```
text
 │
 ├─ word TF-IDF      unigrams + bigrams, min_df=1
 └─ char TF-IDF      char_wb 3-5 grams, min_df=2
 │
 └─ FeatureUnion ──→ LogisticRegression (C=10) ──→ P(positive)
```

The same pipeline is trained twice, once per language. Two deliberate choices drive the accuracy:

- **No stop word list.** The default English list removes *not*, *no*, and *very*, which carry the sentiment in short reviews. The Indonesian equivalents *tidak*, *nggak*, and *banget* matter the same way. Keeping them is worth roughly 8 points.
- **Character n-grams alongside words.** `char_wb` 3-5 grams catch typos and stretched intensifiers such as "terrrible" and "baguuus" that word features miss. Worth roughly 1.5 points.

Held-out accuracy on a stratified 20% split:

| Model | Accuracy | Test size |
| --- | --- | --- |
| English | **0.8467** | 600 |
| Indonesian | **0.9135** | 2,197 |

The Indonesian model scores higher mostly because it has 3.6x the training data. Both are trained on reviews, so casual conversation sits outside what they have seen. Treat low-confidence predictions as uncertain rather than wrong.

### Language detection

`lang.py` counts function words from each language and picks the winner. The two languages share almost no function words, so a vote is accurate enough to route a request and adds no dependency.

A tie, or text with no markers at all, returns `unknown` and the API answers `422`. Guessing there would return a confident-looking score on a sentence neither model was trained for.

```bash
python lang.py   # runs the self-check
```

## Project structure

```
.
├── backend/
│   ├── train.py           Downloads data, trains, saves both models
│   ├── lang.py            Language detection, with a self-check
│   ├── app.py             Flask API: /health and /predict
│   ├── requirements.txt   Pinned Python dependencies
│   └── data/              Dataset cache (gitignored)
├── frontend/
│   ├── src/
│   │   ├── App.jsx        Form, the reactive face, error states
│   │   ├── main.jsx       React entry point
│   │   └── index.css      Tailwind theme tokens
│   ├── .env.example       VITE_API_URL for deployed builds
│   ├── index.html
│   └── vite.config.js
├── render.yaml            API service: build, start, health check
├── netlify.toml           Frontend build and SPA fallback
└── docs/assets/           README screenshot
```

Neither the `.joblib` models nor `backend/data/` is committed. Both are reproducible from `train.py`, and binary artifacts in git make diffs useless.

## Deploying

The two halves go to two hosts. Netlify serves static files only, so the Flask API needs somewhere that runs Python. Render is used here because `render.yaml` describes the whole service, so there is nothing to click through.

### 1. API on Render

Create a **Blueprint** from this repository. `render.yaml` sets the build command, the start command, and the health check path, so the only manual step is the environment variable in step 3.

The models are trained during the build rather than committed, so the first build takes a few minutes. Once it finishes, confirm the API:

```bash
curl https://<your-service>.onrender.com/health
```

The free tier sleeps after 15 minutes of inactivity, so the first request after an idle spell takes roughly 50 seconds while the service wakes. It also caps memory at 512 MB, which is why `render.yaml` runs a single worker: the process plus scikit-learn and both models sits near 177 MB, and a second worker would double that.

### 2. Frontend on Netlify

Connect the repository. `netlify.toml` already sets the base directory, build command, publish directory, and the SPA fallback, so no manual configuration is needed.

Add one environment variable before the first deploy:

| Key | Value |
| --- | --- |
| `VITE_API_URL` | `https://<your-service>.onrender.com` |

Vite inlines this at build time, so changing it later needs a redeploy, not just a restart.

### 3. Allow the Netlify origin

The API rejects unknown origins, so the deployed site cannot call it until you say so. In the Render dashboard add:

| Key | Value |
| --- | --- |
| `ALLOWED_ORIGINS` | `https://<your-site>.netlify.app` |

Several origins can be listed comma separated. Localhost stays allowed either way, so local development keeps working.

## Notes on security

CORS is an allowlist, never a wildcard: `http://localhost:5173` and `http://127.0.0.1:5173` plus whatever `ALLOWED_ORIGINS` contains. A wildcard would let any site drive this API from a visitor's browser.

The API has no authentication or rate limiting. It is a public endpoint running a small CPU model, so the realistic exposure is someone burning your Space quota rather than data loss. Put it behind a proxy with rate limiting if that matters to you.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `Missing model_en.joblib, model_id.joblib` | Run `python train.py` before `python app.py`. |
| `Could not tell which language this is` | The text has no English or Indonesian function words. Pass `"language": "en"` or `"id"` explicitly. |
| `Cannot reach the API` in the UI | The Flask server is not running, or the frontend is not on port 5173. |
| CORS error in the browser console | Serve the frontend on 5173, or add your origin to the `CORS(...)` list in `app.py`. |
| Dataset download fails | Check your connection; `train.py` reuses `backend/data/` if the files are already there. |

## Future improvements

- Neutral class. SmSA already labels it; the English side needs a three-way dataset, and the UI needs its own treatment for it.
- More languages, which the per-language model registry in `app.py` already allows for
- Model comparison against linear SVM and naive Bayes
- Per-word contribution highlighting so a prediction can be inspected
- Deployment: Vercel for the frontend, Render for the API

## License

[MIT](LICENSE)

## Author

Rizky Putra Ichfiandra — AI / ML enthusiast.
