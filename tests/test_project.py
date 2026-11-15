import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.data import audit_and_clean, read_astd, split_data
from src.inference import available_models, load_model, predict
from src.preprocessing import ArabicNormalizer
from src.training import train

PROJECT = Path(__file__).resolve().parents[1]
SENTENCES = {
    "POS": "هذه خدمة ممتازة",
    "NEG": "هذه خدمة سيئة",
    "NEUTRAL": "الأكل حلو لكن التوصيل سيئ",
    "OBJ": "المتجر يفتح الساعة التاسعة",
}


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "synthetic.txt"
    rows = (f"{text} {number}\t{label}"
            for label, text in SENTENCES.items() for number in range(20))
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


@pytest.fixture
def trained(dataset, tmp_path):
    config = json.loads((PROJECT / "config.json").read_text(encoding="utf-8"))
    config["c_values"] = [2.0]
    config["class_weights"] = [None]
    model_dir = tmp_path / "models"
    report_dir = train(dataset, model_dir, tmp_path / "runs", config)
    return model_dir, report_dir


def test_normalization():
    normalize = ArabicNormalizer()
    assert normalize("  ﺇِنَّ الأَكْـلَ  لَذِيذٌ! 😊 @user https://example.org  ") == "ان الاكل لذيذ! 😊"
    assert normalize("لَا أحبُّه ومش هطلبه") == "لا احبه ومش هطلبه"
    custom = ArabicNormalizer(remove_diacritics=False, remove_tatweel=False,
                              keep_punctuation=False, keep_emoji=False)
    assert custom("حَـلو، 😊") == "حَـلو"


def test_deduplication_and_conflicts():
    raw = pd.DataFrame([
        ("أحب الطعام", "positive"), ("احب الطعام", "positive"), ("أحب الطعام", "positive"),
        ("جيد", "positive"), ("جَيِّد", "negative"),
        ("", "negative"), ("https://example.org", "objective"), ("صالح", None),
    ], columns=["text", "label"])
    clean, audit = audit_and_clean(raw, ArabicNormalizer())
    assert clean["normalized"].tolist() == ["احب الطعام"]
    assert audit["exact_duplicate_rows"] == 1
    assert audit["normalized_duplicate_rows"] == 3
    assert audit["conflicting_groups"] == 1
    assert audit["conflicting_rows_removed"] == 2
    assert audit["consistent_duplicates_removed"] == 2
    assert audit["missing_text"] == audit["missing_label"] == audit["empty_after_normalization"] == 1


def test_label_mapping_and_rejection(dataset, tmp_path):
    assert set(read_astd(dataset)["label"]) == {"positive", "negative", "mixed", "objective"}
    malformed = tmp_path / "malformed.txt"
    malformed.write_text("نص\tUNKNOWN\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        read_astd(malformed)


def test_split_integrity(dataset):
    clean, _ = audit_and_clean(read_astd(dataset), ArabicNormalizer())
    splits = split_data(clean)
    assert [len(split) for split in splits.values()] == [56, 12, 12]
    for key in ("text", "normalized"):
        pooled = pd.concat(splits.values())[key]
        assert pooled.is_unique
        assert set(pooled) == set(clean[key])
    for name, split in splits.items():
        assert split.equals(split_data(clean)[name])
        assert split["label"].value_counts().nunique() == 1
    duplicate = pd.concat([clean, clean.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Deduplicate"):
        split_data(duplicate)


def test_training_reports_artifacts_and_prediction(trained):
    models, reports = trained
    winner, files = available_models(models)
    selection = json.loads((reports / "selection.json").read_text())
    evaluation = json.loads((reports / "evaluation.json").read_text())
    assert winner == selection["winner"]
    assert set(files) == {"logistic", "svm"}
    assert "test" not in selection
    assert sum(map(sum, evaluation["test"]["confusion_matrix"])) == 12
    assert (reports / "confusion_matrix.png").is_file()
    for name, path in files.items():
        result = predict(load_model(path), SENTENCES["POS"])
        assert result.label == "positive"
        expected = "Logistic decision score" if name == "logistic" else "SVM decision score"
        assert result.score_name == expected


@pytest.mark.parametrize("text", ["", "  ", None, 123, "hello", "https://example.org", "ي" * 10001, "ضظذ"])
def test_invalid_inference(trained, text):
    model_dir, _ = trained
    winner, files = available_models(model_dir)
    with pytest.raises(ValueError):
        predict(load_model(files[winner]), text)


def test_streamlit_inference(trained, monkeypatch):
    model_dir, _ = trained
    monkeypatch.setenv("PROJECT05_MODELS", str(model_dir))
    app = AppTest.from_file(str(PROJECT / "app.py")).run()
    assert not app.exception and not app.error
    app.button[0].click().run()
    assert app.warning
    app.text_area[0].input(SENTENCES["POS"]).run()
    assert app.text[0].value == SENTENCES["POS"]
    app.button[0].click().run()
    assert not app.exception
    assert app.metric[0].value.startswith("Positive")


def test_streamlit_without_models(tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECT05_MODELS", str(tmp_path))
    app = AppTest.from_file(str(PROJECT / "app.py")).run()
    assert not app.exception
    assert "Train the models" in app.info[0].value
    assert not app.metric
