<div align="center">

# Sentiment Intelligence

**Read the tone before you reply.**

Paste an English review or message. A TF-IDF and logistic regression model scores it as positive or negative and shows how certain it is.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1-000000?logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?logo=scikitlearn&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-8-646CFF?logo=vite&logoColor=white)
![Tailwind](https://img.shields.io/badge/Tailwind-4-06B6D4?logo=tailwindcss&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-3B82F6)

<img src="docs/assets/screenshot.png" alt="Sentiment Intelligence analysing a product review" width="100%" />

</div>

---

## What it does

- **Classifies English text** as positive or negative with a confidence percentage.
- **Shows both class scores**, so a 51/49 split is visibly different from a 95/5 one.
- **Handles negation.** "Not good at all" lands on negative, because the model keeps words like *not*, *no*, and *very* instead of stripping them as stop words.
- **Fails loudly, not silently.** Empty input, oversized input, and an unreachable API each get their own message.

## Built with

| Technology | What it does here |
| --- | --- |
| [scikit-learn](https://scikit-learn.org) | TF-IDF features into logistic regression, wrapped in one `Pipeline`. |
| [Flask](https://flask.palletsprojects.com) + [Flask-CORS](https://flask-cors.readthedocs.io) | Serves `/predict` and `/health` to the browser on localhost. |
| [joblib](https://joblib.readthedocs.io) | Persists the fitted pipeline as a single `model.joblib`. |
| [React](https://react.dev) 19 + [Vite](https://vite.dev) 8 | Interface and dev server with fast rebuilds. |
| [Tailwind CSS](https://tailwindcss.com) 4 | CSS-first theming through `@theme`, no config file. |

## Getting started

### Prerequisites

- Python 3.10 or newer
- Node.js 20 or newer

### 1. Train the model

```bash
cd backend
pip install -r requirements.txt
python train.py
```

`train.py` downloads the [UCI Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences) dataset into `backend/data/`, trains the pipeline, prints a classification report, and writes `model.joblib`. It ends with a self-check that reloads the saved file and asserts two obvious cases, so a broken artifact fails here rather than at request time.

Expected output:

```
3000 labelled sentences  |  positive=1500  negative=1500

Accuracy: 0.8467
...
saved model.joblib
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
{ "status": "ok", "model": "model.joblib" }
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
  "scores": { "positive": 67.7, "negative": 32.3 }
}
```

| Status | When |
| --- | --- |
| `200` | Prediction returned. |
| `400` | `text` missing, not a string, or blank. |
| `413` | `text` longer than 2,000 characters. |

## How the model works

```
text
 │
 ├─ word TF-IDF      unigrams + bigrams, min_df=1
 └─ char TF-IDF      char_wb 3-5 grams, min_df=2
 │
 └─ FeatureUnion ──→ LogisticRegression (C=10) ──→ P(positive)
```

Two deliberate choices drive the accuracy:

- **No stop word list.** The default English list removes *not*, *no*, and *very*, which carry the sentiment in short reviews. Keeping them is worth roughly 8 points.
- **Character n-grams alongside words.** `char_wb` 3-5 grams catch typos and stretched intensifiers such as "terrrible" and "sooo good" that word features miss. Worth roughly 1.5 points.

Held-out accuracy is **0.8467** on a stratified 20% split of 3,000 sentences. Treat low-confidence predictions as uncertain rather than wrong.

## Project structure

```
.
├── backend/
│   ├── train.py           Downloads data, trains, saves model.joblib
│   ├── app.py             Flask API: /health and /predict
│   ├── requirements.txt   Pinned Python dependencies
│   └── data/              Dataset cache (gitignored)
├── frontend/
│   ├── src/
│   │   ├── App.jsx        Form, result card, error states
│   │   ├── main.jsx       React entry point
│   │   └── index.css      Tailwind theme tokens
│   ├── index.html
│   └── vite.config.js
└── docs/assets/           README screenshot
```

Neither `model.joblib` nor `backend/data/` is committed. Both are reproducible from `train.py`, and binary artifacts in git make diffs useless.

## Notes on security

CORS is limited to `http://localhost:5173` and `http://127.0.0.1:5173` for local development. Narrow this to your real origin in `app.py` before deploying anywhere public. The API has no authentication or rate limiting, so do not expose it to the internet as-is.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `model.joblib not found` | Run `python train.py` before `python app.py`. |
| `Cannot reach the API` in the UI | The Flask server is not running, or the frontend is not on port 5173. |
| CORS error in the browser console | Serve the frontend on 5173, or add your origin to the `CORS(...)` list in `app.py`. |
| Dataset download fails | Check your connection; `train.py` reuses `backend/data/` if the files are already there. |

## Future improvements

- Neutral class, which needs a three-way labelled dataset
- Model comparison against linear SVM and naive Bayes
- Per-word contribution highlighting so a prediction can be inspected
- Deployment: Vercel for the frontend, Render for the API

## License

[MIT](LICENSE)

## Author

Rizky Putra Ichfiandra — AI / ML enthusiast.
