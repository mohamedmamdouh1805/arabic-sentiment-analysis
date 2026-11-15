"""Compare word and character features on the original ASTD validation split."""

import argparse
import hashlib
import json
from pathlib import Path

import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from threadpoolctl import threadpool_limits

from src.data import DATA_PATH, RUN_DIR, audit_and_clean, read_astd, split_data
from src.evaluation import evaluate
from src.preprocessing import ArabicNormalizer
from src.training import build_pipeline, write_json


def run(data_path, frozen):
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if digest != frozen["source_sha256"] or sklearn.__version__ != frozen["scikit_learn"]:
        raise ValueError("Use the original dataset and scikit-learn version to recreate the split.")

    config = frozen["config"]
    normalize = ArabicNormalizer(**config["normalization"])
    clean, _ = audit_and_clean(read_astd(data_path), normalize)
    splits = split_data(clean, config["seed"], config["validation_size"], config["test_size"])
    train, validation = splits["train"], splits["validation"]
    for name, frame in (("train", train), ("validation", validation)):
        if frame["label"].value_counts().to_dict() != frozen["splits"][name]:
            raise ValueError(f"The {name} class counts differ from the frozen run.")

    labels = frozen["labels"]
    majority = DummyClassifier(strategy="most_frequent").fit(train["text"], train["label"])
    result = {
        "source_sha256": digest,
        "seed": config["seed"],
        "labels": labels,
        "majority_class": labels[majority.class_prior_.argmax()],
        "majority_validation": evaluate(majority, validation, labels),
        "character_features": {"analyzer": "char_wb", "ngram_range": [3, 5], "min_df": 2},
        "candidates": {},
    }
    with threadpool_limits(limits=1):
        for name, original in frozen["selection"]["best_by_family"].items():
            c, weight = original["C"], original["class_weight"]
            model = build_pipeline(name, normalize, c, weight, config["seed"])
            combined = FeatureUnion([
                ("word", model.named_steps["tfidf"]),
                ("character", TfidfVectorizer(
                    preprocessor=normalize, analyzer="char_wb", ngram_range=(3, 5),
                    min_df=2, sublinear_tf=True,
                )),
            ])
            model.set_params(tfidf=combined)
            model.fit(train["text"], train["label"])
            result["candidates"][name] = {
                "C": c,
                "class_weight": weight,
                "original_validation_macro_f1": original["validation"]["macro"]["f1-score"],
                "validation": evaluate(model, validation, labels),
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--frozen-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=RUN_DIR / "word-char-validation.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists. Choose a new path to preserve existing results.")
    frozen = json.loads(args.frozen_report.read_text(encoding="utf-8"))
    result = run(args.data, frozen)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(f"Validation results saved to {args.output}")


if __name__ == "__main__":
    main()
