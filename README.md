# Arabic Sentiment Analysis

A CPU-trained Arabic sentiment classifier with two scikit-learn baselines and a Streamlit demo. Enter a sentence to see the normalized text, a predicted class, and the selected model’s decision score.

## Dataset

The [Arabic Sentiment Tweets Dataset (ASTD)](https://github.com/mahmoudnabil/ASTD) was published by Mahmoud Nabil, Mohamed Aly, and Amir F. Atiya at EMNLP 2015. It contains manually annotated tweets from an Egyptian Twitter collection. Cite the [original paper](https://aclanthology.org/D15-1299/) when using the dataset. The upstream repository carries a [GPL-2.0 license](https://github.com/mahmoudnabil/ASTD/blob/master/LICENSE). Download the data from upstream. GPL-2.0 §§ 1–3 set conditions for distributing covered material. Preserve applicable copyright, warranty, and license notices and provide a copy of the license when redistributing verbatim copies. Covered modifications must carry change notices and dates and be licensed under the GPL when distributed. GPL-2.0 also has corresponding-source terms for covered executable distributions.

| ASTD code | Class used here | Meaning |
| --- | --- | --- |
| POS | Positive | Subjective positive sentiment |
| NEG | Negative | Subjective negative sentiment |
| NEUTRAL | Mixed | Subjective mixed sentiment, as described in the paper |
| OBJ | Objective | Objective content |

The 1.49 MB source file is verified by SHA-256. The release contains independently written project code and aggregate metrics. ASTD text, upstream code, and trained models are kept out of Git. The [MIT license](LICENSE) applies only to the original project source code. The upstream ASTD terms continue to apply to ASTD material.

## Arabic preprocessing

The same callable runs during training and inference. NFKC folds compatibility forms such as Arabic presentation glyphs. The normalizer maps alef variants to bare alef and alef maqsura to ya, removes tatweel and Arabic diacritics, case-folds Latin text, and collapses whitespace. URLs and mentions are removed to reduce reliance on account and link identity. The defaults preserve punctuation, emojis, and negation such as **لا** and **مش**. Punctuation and emoji handling can be changed in [config.json](config.json) and takes effect after retraining.

TF-IDF uses single tokens and adjacent token pairs. Bigrams can represent combinations such as **مش كويس**. Letter folding reduces spelling variation, with some loss of distinctions. The source was already cleaned by its authors, so support for emoji in preprocessing does not establish that the models learned useful emoji weights.

## Evaluation protocol

The data audit counts missing values, exact duplicates, duplicates after normalization, and normalized groups with conflicting labels. The pipeline drops every row in a conflicting group and keeps one representative from each consistent duplicate group. It then uses seeded, stratified splits targeting 70% training, 15% validation, and 15% test. Normalized text and exact text are disjoint across splits.

Each baseline fits its TF-IDF vocabulary and classifier on training data. The grid compares three values of C and two class-weight options. Validation macro F1 selects the best candidate in each family and the overall winner. Ties preserve the order in the configuration and training code. The training script writes the selection record before scoring the winner once on the held-out test set. The exported pipelines are the same fitted models used for those scores. Validation also supplies up to 200 single-text latency measurements per model after warm-up. Timing covers preprocessing, TF-IDF, and the classifier’s prediction call.

## Original frozen evaluation

The audit found 10,006 source rows, with zero missing texts, missing labels, or texts emptied by normalization. Four exact duplicates were included in the nine excess rows identified by normalized-text matching. Removing two rows from one conflicting group and eight consistent duplicates left 9,996 texts.

| Class | Source | Clean | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mixed | 832 | 832 | 582 | 125 | 125 |
| Negative | 1,684 | 1,683 | 1,179 | 252 | 252 |
| Objective | 6,691 | 6,683 | 4,677 | 1,003 | 1,003 |
| Positive | 799 | 798 | 558 | 120 | 120 |
| **Total** | **10,006** | **9,996** | **6,996** | **1,500** | **1,500** |

| Model and split | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression, validation | 62.87% | 0.423 | 0.432 | **0.427** |
| Linear SVM, validation | 66.93% | 0.451 | 0.406 | 0.421 |
| Logistic regression, held-out test | 60.00% | 0.390 | 0.399 | **0.394** |

Both finalists used balanced class weights. The logistic model used C = 2.0 and fit in 5.34 s. The SVM used C = 0.5 and fit in 0.84 s. Median single-text prediction latency was 0.79 ms for each on the local CPU, with p95 of 1.01 ms and 1.06 ms, respectively.

| Held-out class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Mixed | 0.208 | 0.208 | 0.208 |
| Negative | 0.374 | 0.417 | 0.394 |
| Objective | 0.776 | 0.739 | 0.757 |
| Positive | 0.201 | 0.233 | 0.216 |

The frozen test confusion matrix has actual classes in rows and predictions in columns.

| Actual / predicted | Mixed | Negative | Objective | Positive |
| --- | ---: | ---: | ---: | ---: |
| Mixed | 26 | 38 | 50 | 11 |
| Negative | 36 | 105 | 94 | 17 |
| Objective | 53 | 126 | 741 | 83 |
| Positive | 10 | 12 | 70 | 28 |

Objective accounted for 66.85% of training and 66.87% of the held-out set. Predicting objective for every test row would achieve 66.87% accuracy and 0.200 macro F1. Macro F1 gives each class equal weight and exposes errors that the majority class can obscure in accuracy. The frozen model correctly predicted 26 of 125 mixed tweets and 28 of 120 positive tweets. Objective was the most common error for each minority class.

## Word and character TF-IDF exploration

The training-label majority class was Objective. Predicting it for all 1,500 validation rows gave **66.87% accuracy** and **0.2004 macro F1** across the four labels.

Two exploratory fits added word-boundary character 3–5-grams with minimum document frequency 2 to the existing word unigram and bigram features. Scikit-learn's FeatureUnion concatenated the two TF-IDF representations at equal weights. Both branches used the original Arabic normalizer and sublinear term frequency. The feature vocabularies and classifiers were fit on the original training rows. Each classifier used its original selected C and balanced weights. The shared validation split was the sole scoring set for these candidates.

| Classifier | C | Word-only validation macro F1 | Word + character validation macro F1 | Change | Word + character validation accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 2.0 | 0.4271 | **0.4495** | +0.0225 | 65.47% |
| Linear SVM | 0.5 | 0.4209 | 0.4236 | +0.0027 | 67.13% |

The combined logistic model improved validation F1 across all four classes. Its mixed-class F1 was still 0.230. Adding characters to the SVM lowered its mixed-class F1 from 0.182 to 0.139. The original word-only logistic regression remains the release default. The combined logistic model is a candidate for independent evaluation because the same validation set guided the original selection. The previously examined test set was excluded from this experiment, and the frozen test metrics above apply solely to the original logistic model.

Run the experiment with an existing original baseline evaluation report and an unused output path:

```powershell
.\.venv\Scripts\python.exe -m src.experiment --data datasets/Tweets.txt --frozen-report "runs/<original-run-id>/evaluation.json" --output runs/word-char-validation.json
```

The experiment checks the dataset SHA-256 and scikit-learn version against the supplied report and recreates the original split from its configuration. It writes aggregate validation metrics only.

## Run locally

Use Python 3.12. From the project directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/download_astd.py datasets/Tweets.txt
.\.venv\Scripts\python.exe -m src.training --data datasets/Tweets.txt --models models --runs runs
$env:PROJECT05_MODELS = (Resolve-Path models).Path
.\.venv\Scripts\python.exe -m streamlit run app.py
```

In the portfolio workspace, the training module defaults to the shared `datasets/project05`, `models/project05`, and `runs/project05` directories. For that layout, use `scripts/download_astd.py ../../datasets/project05/Tweets.txt` and omit the explicit path arguments. The app also defaults to the shared model directory there.

Training writes a versioned run directory with `selection.json`, `evaluation.json`, and `confusion_matrix.png`. The model directory contains both fitted pipelines and a manifest that points to the latest complete export. Load joblib files only from a trusted source, using the same scikit-learn version that trained them.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Limitations and future work

The data reflects Egyptian social-media writing from the collection period. The splits measure performance within that corpus, with remaining risks from near-duplicates and recurring topics. Dialect shifts, sarcasm, and domain changes can reduce accuracy on inputs such as product reviews. The UI displays classifier decision scores on their native scales. Future work could confirm the exploratory result on a fresh sample, evaluate on a separate review corpus, and measure calibration before exposing probabilities.
