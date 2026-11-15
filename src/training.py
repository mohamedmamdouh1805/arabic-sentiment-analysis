import argparse
import hashlib
import json
import platform
from collections import Counter
from itertools import product
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from src.data import DATA_PATH, MODEL_DIR, PROJECT, RUN_DIR, audit_and_clean, read_astd, split_data
from src.evaluation import benchmark, evaluate, save_confusion_matrix
from src.preprocessing import ArabicNormalizer, tokenize

MODELS = {"logistic": LogisticRegression, "svm": LinearSVC}


def build_pipeline(name, normalize, c, weight, seed):
    classifier = MODELS[name](C=c, class_weight=weight, max_iter=3000, random_state=seed)
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=normalize, tokenizer=tokenize, token_pattern=None,
            ngram_range=(1, 2), sublinear_tf=True,
        )),
        ("classifier", classifier),
    ])


def fit_candidates(splits, normalize, config, labels):
    train, validation = splits["train"], splits["validation"]
    best_models, best_trials, trials = {}, {}, []
    for name in MODELS:
        for c, weight in product(config["c_values"], config["class_weights"]):
            model = build_pipeline(name, normalize, c, weight, config["seed"])
            start = perf_counter()
            model.fit(train["text"], train["label"])
            seconds = perf_counter() - start
            metrics = evaluate(model, validation, labels)
            trial = {
                "model": name, "C": c, "class_weight": weight, "fit_seconds": seconds,
                "validation": metrics,
            }
            trials.append(trial)
            incumbent = best_trials.get(name)
            if incumbent is None or metrics["macro"]["f1-score"] > incumbent["validation"]["macro"]["f1-score"]:
                best_models[name], best_trials[name] = model, trial
    return best_models, best_trials, trials


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def train(data_path, model_dir, run_dir, config):
    normalize = ArabicNormalizer(**config["normalization"])
    clean, audit = audit_and_clean(read_astd(data_path), normalize)
    splits = split_data(clean, config["seed"], config["validation_size"], config["test_size"])
    labels = sorted(clean["label"].unique())
    if len(labels) < 2:
        raise ValueError("Training requires at least two sentiment classes.")

    run_id = uuid4().hex
    reports = run_dir / run_id
    reports.mkdir(parents=True)
    with threadpool_limits(limits=1):
        models, best, trials = fit_candidates(splits, normalize, config, labels)
        winner = max(best, key=lambda name: best[name]["validation"]["macro"]["f1-score"])
        selection = {
            "winner": winner,
            "criterion": "validation macro F1",
            "best_by_family": best,
            "candidates": trials,
        }
        write_json(reports / "selection.json", selection)
        test = evaluate(models[winner], splits["test"], labels)
        latency = {name: benchmark(model, splits["validation"]["text"].tolist())
                   for name, model in models.items()}

    report = {
        "source_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "config": config,
        "audit": audit,
        "splits": {name: dict(Counter(split["label"])) for name, split in splits.items()},
        "labels": labels,
        "selection": selection,
        "test": test,
        "single_text_latency": latency,
    }
    write_json(reports / "evaluation.json", report)
    save_confusion_matrix(test, labels, reports / "confusion_matrix.png")
    output = model_dir / run_id
    output.mkdir(parents=True)
    for name, model in models.items():
        joblib.dump(model, output / f"{name}.joblib", compress=3)
    pending = model_dir / f"{run_id}.json"
    write_json(pending, {"run_id": run_id, "winner": winner, "scikit_learn": sklearn.__version__})
    pending.replace(model_dir / "latest.json")
    return reports


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Arabic sentiment baselines.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--models", type=Path, default=MODEL_DIR)
    parser.add_argument("--runs", type=Path, default=RUN_DIR)
    parser.add_argument("--config", type=Path, default=PROJECT / "config.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(f"Reports saved to {train(args.data, args.models, args.runs, config)}")


if __name__ == "__main__":
    main()
