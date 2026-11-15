import os
from pathlib import Path

import streamlit as st

from src.data import MODEL_DIR
from src.inference import available_models, load_model, normalized_input, predict

EXAMPLES = [
    "الخدمة ممتازة والتوصيل سريع، شكراً لكم!",
    "للأسف الطلب اتأخر والأكل كان بارد.",
    "الأكل حلو بس التوصيل اتأخر.",
    "يفتح المتجر في الساعة التاسعة صباحاً.",
]
LABELS = {"positive": "إيجابي", "negative": "سلبي", "mixed": "مختلط", "objective": "موضوعي"}
NAMES = {"logistic": "Logistic regression", "svm": "Linear SVM"}


def set_example():
    st.session_state["arabic_text"] = st.session_state["example"]


@st.cache_resource
def cached_model(path):
    return load_model(path)


def main():
    st.set_page_config(page_title="Arabic Sentiment Analysis", page_icon="💬")
    st.markdown("""
    <style>
    .st-key-arabic_text textarea {
        direction: rtl;
        text-align: right;
        font-size: 1.2rem;
        line-height: 1.8;
    }
    </style>
    """, unsafe_allow_html=True)
    st.title("Arabic Sentiment Analysis")
    st.write("Classify an Arabic sentence using a model trained on Egyptian Twitter data.")
    try:
        winner, files = available_models(Path(os.getenv("PROJECT05_MODELS", str(MODEL_DIR))))
    except FileNotFoundError:
        st.info("Train the models first with `python -m src.training`.")
        return
    except (ValueError, KeyError, TypeError, OSError) as error:
        st.error(f"Model configuration is unavailable. {error}")
        return

    model_name = st.selectbox("Model", [winner, *(name for name in files if name != winner)],
                              format_func=NAMES.__getitem__)
    try:
        model = cached_model(files[model_name])
    except Exception:
        st.error("The model could not be loaded. Rebuild it with the installed dependencies.")
        return
    st.selectbox("Try an example", EXAMPLES, index=None, key="example", on_change=set_example)
    text = st.text_area("Arabic text", key="arabic_text", height=140,
                        placeholder="اكتب جملة باللغة العربية هنا")
    submitted = st.button("Analyze sentiment", type="primary")
    if text or submitted:
        try:
            normalized = normalized_input(model, text)
            st.markdown("#### Normalization preview")
            st.text(normalized)
            if submitted:
                result = predict(model, text)
                sentiment, score = st.columns(2)
                sentiment.metric("Sentiment", f"{result.label.title()} · {LABELS[result.label]}")
                score.metric(result.score_name, f"{result.score:.3f}")
                st.caption("Scores use each classifier’s own scale and can be negative. "
                           "Multiclass logistic scores are logits.")
        except ValueError as error:
            st.warning(str(error))

    with st.expander("How the model reads Arabic"):
        st.write("The pipeline normalizes Arabic spelling and whitespace, removes links, mentions, "
                 "diacritics and tatweel, then weighs single tokens and adjacent pairs using TF-IDF.")
        st.write("Mixed means the tweet expresses both positive and negative sentiment. "
                 "Objective means the tweet lacks subjective sentiment.")
    st.caption("Predictions can be unreliable for other dialects, sarcasm, and text outside "
               "the dataset’s social-media domain.")


if __name__ == "__main__":
    main()
