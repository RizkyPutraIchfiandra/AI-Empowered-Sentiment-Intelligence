"""Trains one sentiment classifier per language.

Writes two artifacts per language: a .joblib for the local Flask API, and a
.json of the fitted weights that the deployed frontend scores in the browser.

Run once:  python train.py

English  : UCI Sentiment Labelled Sentences (Amazon, IMDb, Yelp)
Indonesian: SmSA from IndoNLU (Prosa.ai document sentiment)
"""

import io
import json
import pathlib
import sys
import urllib.request
import zipfile

import pandas as pd
import lang
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import joblib

HERE = pathlib.Path(__file__).parent
DATA_DIR = HERE / "data"
WEB_DIR = HERE.parent / "frontend" / "public" / "models"
PARITY_PATH = HERE.parent / "frontend" / "src" / "parity.json"
MARKERS_PATH = HERE.parent / "frontend" / "src" / "markers.json"

EN_URL = "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip"
EN_MEMBERS = ["amazon_cells_labelled.txt", "imdb_labelled.txt", "yelp_labelled.txt"]

SMSA_BASE = "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/smsa_doc-sentiment-prosa"
ID_FILES = ["train_preprocess.tsv", "valid_preprocess.tsv"]


def fetch(url: str, dest: pathlib.Path) -> None:
    if dest.exists():
        return
    print(f"  downloading {dest.name}")
    with urllib.request.urlopen(url, timeout=180) as resp:
        dest.write_bytes(resp.read())


def load_english() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not all((DATA_DIR / m).exists() for m in EN_MEMBERS):
        print(f"  downloading {EN_URL.rsplit('/', 1)[-1]}")
        with urllib.request.urlopen(EN_URL, timeout=180) as resp:
            blob = resp.read()
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            for name in zf.namelist():
                base = pathlib.PurePosixPath(name).name
                if base in EN_MEMBERS:
                    (DATA_DIR / base).write_bytes(zf.read(name))

    missing = [m for m in EN_MEMBERS if not (DATA_DIR / m).exists()]
    if missing:
        sys.exit(f"English dataset incomplete, missing: {', '.join(missing)}")

    frames = [
        pd.read_csv(DATA_DIR / m, sep="\t", header=None, names=["text", "label"], quoting=3)
        for m in EN_MEMBERS
    ]
    df = pd.concat(frames, ignore_index=True).dropna()
    df["text"] = df["text"].astype(str).str.strip()
    return df[df["text"].str.len() > 0]


def load_indonesian() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in ID_FILES:
        fetch(f"{SMSA_BASE}/{name}", DATA_DIR / name)

    frames = [
        pd.read_csv(DATA_DIR / name, sep="\t", header=None, names=["text", "label"], quoting=3)
        for name in ID_FILES
    ]
    df = pd.concat(frames, ignore_index=True).dropna()
    df["text"] = df["text"].astype(str).str.strip()

    # Drop `neutral` to keep the API binary and matched to the English model.
    # A third class needs its own UI treatment, not a silent extra label.
    df = df[df["label"].isin(["positive", "negative"])].copy()
    df["label"] = (df["label"] == "positive").astype(int)
    return df[df["text"].str.len() > 0]


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            # No stop word list on purpose: "not"/"tidak" and "very"/"banget"
            # carry the sentiment, and dropping them costs real accuracy.
            #
            # Words only, no char n-grams. Char features are worth 1.3 points on
            # English and nothing on Indonesian, but they triple the exported
            # vocabulary, and the browser is where this model actually runs.
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("clf", LogisticRegression(C=10.0, max_iter=3000, class_weight="balanced")),
        ]
    )


def export_json(name: str, model: Pipeline) -> pathlib.Path:
    """Write the fitted weights for predict.js to score in the browser.

    Everything the prediction needs is linear: the vocabulary, the idf vector,
    the coefficients, and the intercept. Rounding to 4 decimals costs nothing
    measurable and shrinks the file by about a third.
    """
    tfidf = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]

    WEB_DIR.mkdir(parents=True, exist_ok=True)
    path = WEB_DIR / f"model_{name}.json"
    path.write_text(
        json.dumps(
            {
                "vocab": {term: int(i) for term, i in tfidf.vocabulary_.items()},
                "idf": [round(float(x), 4) for x in tfidf.idf_],
                "coef": [round(float(x), 4) for x in clf.coef_[0]],
                "intercept": round(float(clf.intercept_[0]), 6),
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return path


def train(name: str, df: pd.DataFrame, positive_case: str, negative_case: str) -> float:
    print(f"\n{'=' * 58}\n{name}\n{'=' * 58}")
    print(f"{len(df)} labelled texts  |  positive={int(df.label.sum())}  negative={int((1 - df.label).sum())}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    predicted = model.predict(X_test)
    acc = accuracy_score(y_test, predicted)
    print(f"\nAccuracy: {acc:.4f}\n")
    print(classification_report(y_test, predicted, target_names=["negative", "positive"]))

    path = HERE / f"model_{name}.joblib"
    joblib.dump(model, path)
    print(f"saved {path.name}")

    # Self-check: the saved artifact must round-trip and agree on clear cases.
    reloaded = joblib.load(path)
    assert reloaded.predict([positive_case])[0] == 1, f"{name}: positive case misclassified"
    assert reloaded.predict([negative_case])[0] == 0, f"{name}: negative case misclassified"
    print("self-check passed")

    exported = export_json(name, model)
    size_mb = exported.stat().st_size / 1024 / 1024
    print(f"saved {exported.parent.name}/{exported.name} ({size_mb:.2f} MB)")

    # Record what scikit-learn predicts for these cases so predict.check.mjs can
    # assert the browser implementation agrees. Without this the JS tokenizer can
    # drift from the vectorizer and nothing would notice.
    PARITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    parity = json.loads(PARITY_PATH.read_text(encoding="utf-8")) if PARITY_PATH.exists() else {}
    parity[name] = [
        {"text": t, "positive": round(float(model.predict_proba([t])[0][1]), 6)}
        for t in (positive_case, negative_case)
    ]
    PARITY_PATH.write_text(json.dumps(parity, indent=2) + "\n", encoding="utf-8")

    return acc


def main() -> None:
    train(
        "en",
        load_english(),
        "this is wonderful, i love it",
        "terrible, broke after one day",
    )
    train(
        "id",
        load_indonesian(),
        "makanan nya enak banget, pelayanan ramah, pasti balik lagi",
        "pelayanan nya lambat, makanan dingin, kecewa banget",
    )

    # The marker lists live in lang.py. Exporting them keeps the browser and the
    # API detecting the same way instead of two lists drifting apart.
    MARKERS_PATH.write_text(
        json.dumps({"id": sorted(lang.ID_MARKERS), "en": sorted(lang.EN_MARKERS)}, indent=0),
        encoding="utf-8",
    )
    print(f"saved {MARKERS_PATH.name}")

    print(f"\nBoth models saved in {HERE}")


if __name__ == "__main__":
    main()
