"""Trains the sentiment classifier and saves it next to this file.

Run once before app.py:  python train.py
"""

import io
import pathlib
import sys
import urllib.request
import zipfile

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

import joblib

HERE = pathlib.Path(__file__).parent
DATA_DIR = HERE / "data"
MODEL_PATH = HERE / "model.joblib"

DATASET_URL = "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip"
# The UCI archive holds three labelled files, one per source site.
MEMBERS = ["amazon_cells_labelled.txt", "imdb_labelled.txt", "yelp_labelled.txt"]


def download() -> None:
    """Fetch and unpack the dataset once; later runs reuse ./data."""
    if all((DATA_DIR / m).exists() for m in MEMBERS):
        print(f"dataset already present in {DATA_DIR}")
        return

    print(f"downloading {DATASET_URL}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DATASET_URL, timeout=120) as resp:
        blob = resp.read()

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for name in zf.namelist():
            base = pathlib.PurePosixPath(name).name
            if base in MEMBERS:
                (DATA_DIR / base).write_bytes(zf.read(name))
                print(f"  extracted {base}")

    missing = [m for m in MEMBERS if not (DATA_DIR / m).exists()]
    if missing:
        sys.exit(f"dataset incomplete, missing: {', '.join(missing)}")


def load() -> pd.DataFrame:
    frames = [
        pd.read_csv(DATA_DIR / m, sep="\t", header=None, names=["text", "label"], quoting=3)
        for m in MEMBERS
    ]
    df = pd.concat(frames, ignore_index=True).dropna()
    df["text"] = df["text"].astype(str).str.strip()
    return df[df["text"].str.len() > 0]


def main() -> None:
    download()
    df = load()
    print(f"\n{len(df)} labelled sentences  |  positive={int(df.label.sum())}  negative={int((1 - df.label).sum())}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    # Word + bigram TF-IDF into logistic regression: small, fast, and it exposes
    # calibrated-enough probabilities for the confidence bar in the UI.
    model = Pipeline(
        [
            # No stop word list on purpose: "not", "no", and "very" carry the
            # sentiment here, and dropping them costs ~8 points of accuracy.
            # Char n-grams on top of words catch typos and intensifiers
            # ("terrrible", "sooo good") that word features alone miss.
            (
                "tfidf",
                FeatureUnion(
                    [
                        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
                        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
                    ]
                ),
            ),
            ("clf", LogisticRegression(C=10.0, max_iter=3000)),
        ]
    )
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"\nAccuracy: {acc:.4f}\n")
    print(classification_report(y_test, model.predict(X_test), target_names=["negative", "positive"]))

    joblib.dump(model, MODEL_PATH)
    print(f"saved {MODEL_PATH.name}")

    # Self-check: the saved artifact must round-trip and agree on obvious cases.
    reloaded = joblib.load(MODEL_PATH)
    assert reloaded.predict(["this is wonderful, i love it"])[0] == 1, "positive case misclassified"
    assert reloaded.predict(["terrible, broke after one day"])[0] == 0, "negative case misclassified"
    print("self-check passed")


if __name__ == "__main__":
    main()
