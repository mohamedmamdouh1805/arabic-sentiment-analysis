import json
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import joblib
import numpy as np
import sklearn
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_is_fitted

from src.preprocessing import has_arabic

MAX_INPUT = 10_000


@dataclass(frozen=True)
class Prediction:
    label: str
    score: float
    score_name: str


def available_models(model_dir):
    manifest = json.loads((model_dir / "latest.json").read_text(encoding="utf-8"))
    run_id = UUID(manifest["run_id"]).hex
    if manifest["scikit_learn"] != sklearn.__version__:
        raise ValueError("Use the scikit-learn version that trained these models.")
    files = {
        name: model_dir / run_id / f"{name}.joblib" for name in ("logistic", "svm")
    }
    files = {name: path for name, path in files.items() if path.is_file()}
    if manifest["winner"] not in files:
        raise ValueError("The selected model file is missing. Run training again.")
    return manifest["winner"], files


def load_model(path: Path):
    # Joblib deserialization is limited to trusted, locally trained artifacts.
    model = joblib.load(path)
    check_is_fitted(model)
    return model


def normalized_input(model, text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter some Arabic text.")
    if len(text) > MAX_INPUT:
        raise ValueError(f"Enter up to {MAX_INPUT:,} characters.")
    vectorizer = model.named_steps["tfidf"]
    normalized = vectorizer.preprocessor(text)
    if not has_arabic(normalized):
        raise ValueError("Include at least one Arabic letter outside links and mentions.")
    return normalized


def predict(model, text):
    normalized_input(model, text)
    features = model.named_steps["tfidf"].transform([text])
    if features.nnz == 0:
        raise ValueError("This text has no features in the model vocabulary. Try a fuller sentence.")
    classifier = model.named_steps["classifier"]
    label = classifier.predict(features)[0]
    index = np.flatnonzero(model.classes_ == label)[0]
    decision = classifier.decision_function(features)
    score = decision[0, index] if decision.ndim == 2 else abs(decision[0])
    name = "SVM decision score" if isinstance(classifier, LinearSVC) else "Logistic decision score"
    return Prediction(label, float(score), name)
