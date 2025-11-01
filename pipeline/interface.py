"""Streamlit interface for the MetaMind sentiment demo."""

import json
from typing import Any, Dict, List

import streamlit as st

from analysis.sentiment import metamind_sentiment_json
from baselines.raw_sentiment import baseline_sentiment_json
from config import LLM_CONFIG
from llm_interface import OpenAILLM


def ensure_api_key(config: Dict[str, Any]) -> None:
    api_key = config.get("api_key")
    if not api_key or api_key in {"your_api_key_here", "replace_me"}:
        raise ValueError(
            "OpenAI API key missing. Set OPENAI_API_KEY or update config.py."
        )


@st.cache_resource
def load_llm() -> OpenAILLM:
    ensure_api_key(LLM_CONFIG)
    return OpenAILLM(LLM_CONFIG)


def render_results(
    baseline_result: Dict[str, Any], metamind_result: Dict[str, Any]
) -> None:
    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("Baseline output")
        st.json(baseline_result)
    with right_col:
        st.subheader("MetaMind output")
        st.json(metamind_result)


def main() -> None:
    st.set_page_config(page_title="MetaMind Sentiment Demo", layout="wide")
    st.title("MetaMind Sentiment Demo")
    st.write(
        "Enter a user review and compare the baseline sentiment output "
        "against the MetaMind enriched result."
    )

    llm = load_llm()
    review = st.text_area(
        "User review",
        placeholder="Paste or type the review you want to analyze.",
        height=160,
    ).strip()

    analyze_clicked = st.button("Analyze sentiment", type="primary")

    if analyze_clicked:
        if not review:
            st.warning("Please enter a review before running the analysis.")
            return

        context: List[Dict[str, str]] = []
        hypotheses: List[Dict[str, Any]] = []

        try:
            with st.spinner("Running sentiment analysis..."):
                baseline_result = baseline_sentiment_json(
                    llm, review, context, max_retries=1
                )
                metamind_result = metamind_sentiment_json(
                    llm, review, context, hypotheses
                )
        except Exception as exc:
            st.error(f"Error while running analysis: {exc}")
            return

        st.success("Analysis complete.")
        render_results(baseline_result, metamind_result)


if __name__ == "__main__":
    main()
