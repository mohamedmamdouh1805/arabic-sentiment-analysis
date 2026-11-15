from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = {"POS": "positive", "NEG": "negative", "NEUTRAL": "mixed", "OBJ": "objective"}
PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1] if PROJECT.parent.name == "projects" else PROJECT
DATA_PATH = ROOT / "datasets" / ("project05" if ROOT != PROJECT else "") / "Tweets.txt"
MODEL_DIR = ROOT / "models" / ("project05" if ROOT != PROJECT else "")
RUN_DIR = ROOT / "runs" / ("project05" if ROOT != PROJECT else "")


def read_astd(path):
    rows = []
    lines = path.read_text(encoding="utf-8-sig").removesuffix("\n").split("\n")
    for number, line in enumerate(lines, 1):
        if not line.strip():
            rows.append(("", None))
            continue
        text, separator, label = line.rstrip("\r").rpartition("\t")
        if not separator or label.strip() not in LABELS:
            raise ValueError(f"Unknown or missing ASTD label on line {number}.")
        rows.append((text, LABELS[label.strip()]))
    return pd.DataFrame(rows, columns=["text", "label"])


def audit_and_clean(frame, normalize):
    valid_labels = frame["label"].dropna()
    if not set(valid_labels).issubset(LABELS.values()):
        raise ValueError("Unexpected sentiment label.")
    missing_text = ~frame["text"].map(lambda text: isinstance(text, str) and bool(text.strip()))
    missing_label = frame["label"].isna()
    clean = frame.loc[~(missing_text | missing_label)].copy()
    clean["normalized"] = clean["text"].map(normalize)
    empty_normalized = clean["normalized"].eq("")
    clean = clean.loc[~empty_normalized]
    conflicts = clean.groupby("normalized")["label"].transform("nunique").gt(1)
    consistent = clean.loc[~conflicts]
    unique = consistent.drop_duplicates("normalized").reset_index(drop=True)
    audit = {
        "raw_rows": len(frame),
        "raw_class_counts": dict(Counter(valid_labels)),
        "missing_text": int(missing_text.sum()),
        "missing_label": int(missing_label.sum()),
        "empty_after_normalization": int(empty_normalized.sum()),
        "exact_duplicate_rows": int(clean["text"].duplicated().sum()),
        "normalized_duplicate_rows": int(clean["normalized"].duplicated().sum()),
        "conflicting_groups": int(clean.loc[conflicts, "normalized"].nunique()),
        "conflicting_rows_removed": int(conflicts.sum()),
        "consistent_duplicates_removed": len(consistent) - len(unique),
        "clean_rows": len(unique),
        "clean_class_counts": dict(Counter(unique["label"])),
    }
    return unique, audit


def split_data(frame, seed=42, validation_size=0.15, test_size=0.15):
    if not (0 < validation_size and 0 < test_size and validation_size + test_size < 1):
        raise ValueError("Validation and test fractions must be positive and sum to less than one.")
    if frame["normalized"].duplicated().any() or frame["text"].duplicated().any():
        raise ValueError("Deduplicate exact and normalized text before splitting.")
    development, test = train_test_split(
        frame, test_size=test_size, stratify=frame["label"], random_state=seed
    )
    train, validation = train_test_split(
        development, test_size=validation_size / (1 - test_size),
        stratify=development["label"], random_state=seed,
    )
    splits = {"train": train, "validation": validation, "test": test}
    if any(set(split["label"]) != set(frame["label"]) for split in splits.values()):
        raise ValueError("Every split must contain every label. Check class counts and fractions.")
    return splits
