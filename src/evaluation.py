from time import perf_counter

import matplotlib
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def evaluate(model, frame, labels):
    predicted = model.predict(frame["text"])
    report = classification_report(
        frame["label"], predicted, labels=labels, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(frame["label"], predicted, labels=labels)
    return {
        "accuracy": accuracy_score(frame["label"], predicted),
        "macro": report["macro avg"],
        "per_class": {label: report[label] for label in labels},
        "confusion_matrix": matrix.tolist(),
    }


def benchmark(model, texts):
    batch = texts[: min(len(texts), 200)]
    model.predict(batch)
    timings = []
    for text in batch:
        start = perf_counter()
        model.predict([text])
        timings.append(1000 * (perf_counter() - start))
    return {
        "samples": len(timings),
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
    }


def save_confusion_matrix(metrics, labels, path):
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    display = ConfusionMatrixDisplay(np.asarray(metrics["confusion_matrix"]), display_labels=labels)
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Held-out test confusion matrix")
    fig.savefig(path, dpi=150)
    plt.close(fig)
