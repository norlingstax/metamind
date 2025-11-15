import json
from pathlib import Path
from typing import Any, Dict

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from wordcloud import WordCloud

from analysis.recommendation_text import recommendation_text_from_result
from analysis.sentiment import metamind_sentiment_json
from baselines.raw_sentiment import baseline_sentiment_json
from config import DATASET_CONFIG, LLM_CONFIG
from llm_interface.openai_llm import OpenAILLM
from utils.dataset_utils import get_dataset_paths, load_reviews_dataframe

DATASET_PATH, DEEPDIVE_PATH = get_dataset_paths()
DEEPDIVE_ENABLED = DATASET_CONFIG.get("deepdive_enabled", False)


def ensure_api_key(config: Dict[str, Any]) -> None:
    api_key = config.get("api_key")
    if not api_key or api_key in {"your_api_key_here", "replace_me"}:
        raise ValueError("OpenAI API key missing. Set OPENAI_API_KEY or update config.py.")


@st.cache_resource
def load_llm() -> OpenAILLM:
    ensure_api_key(LLM_CONFIG)
    return OpenAILLM(LLM_CONFIG)


@st.cache_data
def load_deep_dive_data(path: str) -> pd.DataFrame:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
    except FileNotFoundError:
        st.error(f"Deep-dive cache not found at {path}.")
        st.stop()
    except json.JSONDecodeError:
        st.error(f"The file '{path}' is corrupt or incomplete.")
        st.stop()

    df = pd.DataFrame(rows)
    if "baseline_result" not in df.columns or "metamind_result" not in df.columns:
        st.error(f"The file '{path}' is missing required fields.")
        st.stop()
    return df


@st.cache_data
def get_processed_data(cache_path: str) -> pd.DataFrame:
    df_raw = load_deep_dive_data(cache_path)
    processed_rows = []

    def get_metamind_top_hyp_type(metamind_res):
        try:
            hypotheses = metamind_res.get("domain_hypotheses", metamind_res.get("hypotheses", []))
            if hypotheses:
                top_hyp = sorted(hypotheses, key=lambda h: h.get("score", 0.0), reverse=True)[0]
                return top_hyp.get("type", "Unknown")
        except Exception:
            pass
        return "N/A"

    def get_baseline_keywords(baseline_res, aspect_name):
        keywords = []
        try:
            for asp in baseline_res.get("aspects", []):
                if isinstance(asp, dict):
                    asp_key = asp.get("name", asp.get("aspect"))
                    if asp_key and asp_key.lower() == aspect_name.lower():
                        evidence_words = str(asp.get("evidence", "")).split()
                        keywords.extend(evidence_words)
        except Exception:
            pass
        return keywords

    for _, row in df_raw.iterrows():
        metamind_res = row.get("metamind_result", {})
        baseline_res = row.get("baseline_result", {})
        review_top_hyp_type = get_metamind_top_hyp_type(metamind_res)
        metamind_aspects = metamind_res.get("aspects", [])

        if not metamind_aspects:
            continue

        for aspect in metamind_aspects:
            aspect_name, aspect_sentiment = "Unknown", "neutral"
            if isinstance(aspect, dict):
                aspect_name = aspect.get("name", aspect.get("aspect", "Unknown")).strip().capitalize()
                aspect_sentiment = aspect.get("sentiment", "neutral")
            elif isinstance(aspect, str):
                aspect_name = aspect.strip().capitalize()
                aspect_sentiment = "neutral"
            else:
                continue

            baseline_keywords = get_baseline_keywords(baseline_res, aspect_name)
            processed_rows.append({
                "review_text": row.get("review_text", ""),
                "aspect_name": aspect_name,
                "aspect_sentiment": aspect_sentiment,
                "review_top_hyp_type": review_top_hyp_type,
                "baseline_keywords": " ".join(baseline_keywords)
            })
    return pd.DataFrame(processed_rows)


@st.cache_data
def get_review_level_comparison_data(cache_path: str) -> pd.DataFrame:
    df_raw = load_deep_dive_data(cache_path)
    comparison_rows = []
    for index, row in df_raw.iterrows():
        review_text = row.get("review_text")
        baseline_res = row.get("baseline_result", {})
        metamind_res = row.get("metamind_result", {})

        if not isinstance(baseline_res, dict) or not isinstance(metamind_res, dict):
            continue

        baseline_polarity = baseline_res.get("polarity", "n/a").lower()
        metamind_polarity = metamind_res.get("polarity", "n/a").lower()

        comparison_rows.append({
            "Index": index,
            "Review": review_text,
            "Baseline Sentiment": baseline_polarity,
            "MetaMind Sentiment": metamind_polarity
        })
    return pd.DataFrame(comparison_rows)


@st.cache_data
def load_reviews_table(csv_path: str) -> pd.DataFrame:
    df = load_reviews_dataframe(Path(csv_path))
    return df.dropna().reset_index(drop=True)


def display_kpi_dashboard(df_comparison: pd.DataFrame, df_processed: pd.DataFrame) -> None:
    st.header("Executive Summary")
    st.markdown("This dashboard provides a high-level summary of the analysis.")

    total_reviews = len(df_comparison)
    if total_reviews == 0 or df_processed.empty:
        st.info("Deep dive cache is empty; skipping KPI dashboard.")
        return

    df_anomalies = df_comparison[
        df_comparison["Baseline Sentiment"] != df_comparison["MetaMind Sentiment"]
    ].copy()
    total_anomalies = len(df_anomalies)
    correction_rate = (total_anomalies / total_reviews) * 100 if total_reviews else 0

    top_driver = df_processed['review_top_hyp_type'].value_counts().index[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews Analyzed", f"{total_reviews}")
    col2.metric("Baseline Errors Corrected", f"{total_anomalies}")
    col3.metric("Data Quality Lift (ROI)", f"{correction_rate:.1f}%")
    col4.metric("Top Customer Driver", top_driver)


def display_deep_dive_section(df_processed: pd.DataFrame) -> None:
    if df_processed.empty:
        st.info("No deep dive data available.")
        return

    st.header("Deep Dive: Aspect-Based Analysis")
    st.markdown("This section provides a detailed breakdown of the most negative customer aspects.")

    neg_aspects = df_processed[df_processed['aspect_sentiment'] == 'negative']
    if neg_aspects.empty:
        st.info("No negative aspects detected in the cache.")
        return

    top_5_neg_aspects = (
        neg_aspects['aspect_name']
        .value_counts()
        .nlargest(5)
        .sort_values(ascending=False)
        .reset_index()
    )
    top_5_neg_aspects.columns = ['aspect', 'count']

    st.subheader("Top 5 Negative Customer Pain Points")

    chart_neg = (
        alt.Chart(top_5_neg_aspects)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusBottomLeft=4)
        .encode(
            x=alt.X('count:Q', title='Number of Mentions'),
            y=alt.Y('aspect:N', sort='-x', title='Aspect'),
            color=alt.Color('aspect:N', legend=None),
            tooltip=[alt.Tooltip('aspect:N', title='Aspect'), alt.Tooltip('count:Q', title='Count')]
        )
        .properties(height=400)
    )

    st.altair_chart(chart_neg, use_container_width=True)

    selected_aspect = st.selectbox(
        "Select a pain point for detailed analysis",
        top_5_neg_aspects['aspect'],
        index=0
    )
    if not selected_aspect:
        return

    st.subheader(f"Comparison: Baseline vs. MetaMind for '{selected_aspect}'")
    aspect_data = df_processed[df_processed['aspect_name'] == selected_aspect]
    if aspect_data.empty:
        st.info("No records available for the selected aspect.")
        return

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("##### Baseline: Keyword Cloud")
            text = " ".join(aspect_data['baseline_keywords'])
            if text.strip():
                try:
                    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                except Exception:
                    st.text("Unable to generate the word cloud.")
            else:
                st.info("No keywords found from Baseline analysis.")

    with col2:
        with st.container(border=True):
            st.markdown("##### MetaMind: State Spirit")
            mental_state_data = (
                aspect_data['review_top_hyp_type']
                .value_counts()
                .sort_values(ascending=False)
                .reset_index()
            )
            mental_state_data.columns = ['state', 'count']

            if not mental_state_data.empty:
                chart = (
                    alt.Chart(mental_state_data)
                    .mark_bar(cornerRadiusTopRight=8, cornerRadiusBottomRight=8)
                    .encode(
                        x=alt.X('count:Q', title='Number'),
                        y=alt.Y('state:N', sort=None, title='Mental State'),
                        color=alt.Color('count:Q', scale=alt.Scale(scheme='reds'), legend=None),
                        tooltip=[alt.Tooltip('state:N', title='State'), alt.Tooltip('count:Q', title='Count')]
                    )
                    .properties(width=800, height=400)
                )
                st.altair_chart(chart, use_container_width=True)

                dominant_state = mental_state_data.iloc[0]['state']
                st.markdown(
                    f"**Insight:** When users discuss **{selected_aspect}**, their dominant state is **{dominant_state}**."
                )
            else:
                st.info("No MetaMind data found for this aspect.")


def display_anomaly_section(df_comparison: pd.DataFrame) -> None:
    if df_comparison.empty:
        st.info("No comparison data available.")
        return

    st.header("Anomaly & Sarcasm Detector")
    st.markdown("This table shows all the reviews for which the Baseline and MetaMind analyses disagreed.")

    df_anomalies = df_comparison[
        df_comparison["Baseline Sentiment"] != df_comparison["MetaMind Sentiment"]
    ].copy()
    df_anomalies = df_anomalies[
        (df_anomalies["Baseline Sentiment"] != "n/a") &
        (df_anomalies["MetaMind Sentiment"] != "n/a")
    ]

    if df_anomalies.empty:
        st.info("No anomalies or disagreements found.")
        return

    st.subheader(f"Analysis of {len(df_anomalies)} disagreements found:")
    gb = GridOptionsBuilder.from_dataframe(df_anomalies)
    cell_style_centered = {'textAlign': 'center'}
    gb.configure_column("Index", flex=0.5, cellStyle=cell_style_centered)
    gb.configure_column("Review", flex=3, wrapText=True, autoHeight=True)
    gb.configure_column("Baseline Sentiment", flex=1, cellStyle=cell_style_centered)
    gb.configure_column("MetaMind Sentiment", flex=1, cellStyle=cell_style_centered)
    grid_options = gb.build()
    AgGrid(
        df_anomalies,
        gridOptions=grid_options,
        height=400,
        width='100%',
        theme="streamlit",
        fit_columns_on_grid_load=True
    )


def main() -> None:
    st.set_page_config(page_title="MetaMind Sentiment Demo", layout="wide")
    st.title(" MetaMind Sentiment Demo")

    df_processed = pd.DataFrame()
    df_comparison = pd.DataFrame()
    show_deep_dive = False
    deepdive_file = str(DEEPDIVE_PATH)
    deepdive_cache_exists = DEEPDIVE_PATH.exists()

    if DEEPDIVE_ENABLED and deepdive_cache_exists:
        try:
            df_processed = get_processed_data(deepdive_file)
            df_comparison = get_review_level_comparison_data(deepdive_file)
            display_kpi_dashboard(df_comparison, df_processed)
            show_deep_dive = True
        except Exception as exc:
            st.error(f"Failed to load dashboard data: {exc}")
            st.info("Live Analysis Only mode. Pre-computed dashboards are disabled.")
    elif DEEPDIVE_ENABLED:
        st.info(
            "Full dataset deep-dive is enabled, but no cache was found yet. "
            f"Run `python -m analysis.precompute_deepdive` to generate "
            f"{DEEPDIVE_PATH.name!s} for {DATASET_PATH.name!s}, then refresh the page."
        )
    else:
        st.info(
            "Full dataset deep-dive analysis is disabled. "
            "Set DATASET_CONFIG['deepdive_enabled']=True to view the aggregate dashboards."
        )

    st.divider()
    st.header("Individual Analysis (Live Test)")
    st.markdown("Select any review in the table below to run a real-time analysis.")

    try:
        df_csv = load_reviews_table(str(DATASET_PATH))
    except FileNotFoundError:
        st.error(f"Could not load the reviews CSV file from path: {DATASET_PATH}")
        st.stop()
    except Exception as exc:
        st.error(f"An error occurred loading {DATASET_PATH}: {exc}")
        st.stop()

    gb = GridOptionsBuilder.from_dataframe(df_csv)
    gb.configure_selection("single", use_checkbox=False)
    grid_options = gb.build()

    grid_response = AgGrid(
        df_csv,
        gridOptions=grid_options,
        height=250,
        theme="streamlit",
    )

    selected_rows = grid_response.get("selected_rows")
    if selected_rows is None:
        selected_rows = []

    if isinstance(selected_rows, pd.DataFrame):
        has_selection = not selected_rows.empty
        selected_review = selected_rows.iloc[0]["Review"] if has_selection else None
    else:
        has_selection = bool(selected_rows)
        selected_review = selected_rows[0]["Review"] if has_selection else None

    if has_selection and selected_review:
        if st.button("Analyze sentiment", type="primary"):
            llm = load_llm()
            context = []
            hypotheses = []
            with st.spinner("Running sentiment analysis..."):
                baseline_result = baseline_sentiment_json(llm, selected_review, context, max_retries=1)
                metamind_result = metamind_sentiment_json(llm, selected_review, context, hypotheses)
                recommendation_text = recommendation_text_from_result(metamind_result)

            left, right = st.columns(2)
            with left:
                st.subheader("Baseline output")
                st.json(baseline_result)
            with right:
                st.subheader("MetaMind output")
                st.json(metamind_result)
                st.subheader("Recommendation")
                st.write(recommendation_text)

            st.success("Analysis complete")
    else:
        st.info("Click on a review in the table to select it.")

    if show_deep_dive:
        st.divider()
        display_deep_dive_section(df_processed)
        st.divider()
        display_anomaly_section(df_comparison)


if __name__ == "__main__":
    main()
