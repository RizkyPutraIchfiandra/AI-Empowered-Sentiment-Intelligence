"""Trains one sentiment classifier per language and saves them next to this file.

Run once before app.py:  python train.py

English  : UCI Sentiment Labelled Sentences (Amazon, IMDb, Yelp)
Indonesian: SmSA from IndoNLU (Prosa.ai document sentiment)
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
            # Char n-grams on top of words catch typos and intensifiers
            # ("terrrible", "baguuus") that word features alone miss.
            (
                "tfidf",
                FeatureUnion(
                    [
                        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
                        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
                    ]
                ),
            ),
            ("clf", LogisticRegression(C=10.0, max_iter=3000, class_weight="balanced")),
        ]
    )


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
    print(f"\nBoth models saved in {HERE}")


if __name__ == "__main__":
    main()
