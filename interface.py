import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from typing import Any, Dict
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder

# --- Imports (Mise à jour pour inclure DATASET_CONFIG) ---
from baselines.raw_sentiment import baseline_sentiment_json
from config import LLM_CONFIG, DATASET_CONFIG  # DATASET_CONFIG est nécessaire pour le CSV
from llm_interface.openai_llm import OpenAILLM
import pandas as pd
from analysis.sentiment import metamind_sentiment_json
from analysis.recommendation_text import recommendation_text_from_result


# --- Fonctions load_llm de votre ami (Inchangées) ---
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


# === DÉBUT : Fonctions de Dashboard pré-calculé (Inchangées) ===

@st.cache_data
def load_deep_dive_data(path: str) -> pd.DataFrame:
    """Loads the pre-computed JSONL data into a DataFrame."""
    try:
        with open(path, 'r') as f:
            data = [json.loads(line) for line in f]
        df = pd.DataFrame(data)
        if "baseline_result" not in df.columns or "metamind_result" not in df.columns:
            st.error(f"Error: The file '{path}' is corrupt or incomplete.")
            st.stop()
        return df
    except FileNotFoundError:
        st.error(f"Error: 'Deep Dive' file ({path}) not found.")
        st.info("Please run the `python precompute_deepdive.py` script first.")
        st.stop()
    except json.JSONDecodeError:
        st.error(f"Error: Could not read file '{path}'. Is it a valid JSONL format?")
        st.stop()


@st.cache_data
def get_processed_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the raw DataFrame into a "clean" table for analysis:
    one row per aspect, per review.
    """
    processed_rows = []

    def get_metamind_top_hyp_type(metamind_res):
        try:
            hypotheses = metamind_res.get("domain_hypotheses", metamind_res.get("hypotheses", []))
            if hypotheses:
                top_hyp = sorted(hypotheses, key=lambda h: h.get('score', 0.0), reverse=True)[0]
                return top_hyp.get('type', 'Unknown')
        except Exception:
            pass
        return 'N/A'

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
                "review_text": row["review_text"],
                "aspect_name": aspect_name,
                "aspect_sentiment": aspect_sentiment,
                "review_top_hyp_type": review_top_hyp_type,
                "baseline_keywords": " ".join(baseline_keywords)
            })
    return pd.DataFrame(processed_rows)


@st.cache_data
def get_review_level_comparison_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the raw DataFrame into a simple table for anomaly detection.
    """
    comparison_rows = []
    for index, row in df_raw.iterrows():
        review_text = row.get("review_text")
        baseline_res = row.get("baseline_result", {})
        metamind_res = row.get("metamind_result", {})

        if not isinstance(baseline_res, dict) or not isinstance(metamind_res, dict):
            continue

        baseline_polarity = baseline_res.get("polarity", "N/A").lower()
        metamind_polarity = metamind_res.get("polarity", "N/A").lower()

        comparison_rows.append({
            "Index": index,
            "Review": review_text,
            "Baseline Sentiment": baseline_polarity,
            "MetaMind Sentiment": metamind_polarity
        })
    return pd.DataFrame(comparison_rows)


def display_kpi_dashboard(df_comparison: pd.DataFrame, df_processed: pd.DataFrame):
    """
    Displays a top-level summary dashboard with key metrics.
    """
    st.header("Executive Summary")
    st.markdown("This dashboard provides a high-level summary of the analysis...")

    total_reviews = len(df_comparison)
    df_anomalies = df_comparison[
        df_comparison["Baseline Sentiment"] != df_comparison["MetaMind Sentiment"]
        ].copy()
    df_anomalies = df_anomalies[
        (df_anomalies["Baseline Sentiment"] != "n/a") &
        (df_anomalies["MetaMind Sentiment"] != "n/a")
        ]
    total_anomalies = len(df_anomalies)

    if total_reviews > 0:
        correction_rate = (total_anomalies / total_reviews) * 100
    else:
        correction_rate = 0

    try:
        top_driver = df_processed['review_top_hyp_type'].value_counts().index[0]
    except IndexError:
        top_driver = "N/A"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews Analyzed", f"{total_reviews}")
    col2.metric("Baseline Errors Corrected", f"{total_anomalies}")
    col3.metric("Data Quality Lift (ROI)", f"{correction_rate:.1f}%")
    col4.metric("Top Customer Driver", top_driver)


def display_deep_dive_section(df_processed: pd.DataFrame):
    """
    Displays the complete "Deep Dive" section.
    """
    st.header("Deep Dive: Aspect-Based Analysis")
    st.markdown("This section provides a deep dive into specific product aspects...")

    neg_aspects = df_processed[df_processed['aspect_sentiment'] == 'negative']
    if neg_aspects.empty:
        st.info("No negative aspects found in the pre-computed data.")
        return

    top_5_neg_aspects = neg_aspects['aspect_name'].value_counts().nlargest(5)
    st.subheader("Top 5 Customer Pain Points (Negative Aspects)")
    st.bar_chart(top_5_neg_aspects)

    selected_aspect = st.selectbox("Select a pain point for a 'Deep Dive'", top_5_neg_aspects.index)
    if not selected_aspect:
        return

    st.subheader(f"Comparison: Baseline vs. MetaMind for '{selected_aspect}'")
    aspect_data = df_processed[df_processed['aspect_name'] == selected_aspect]
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Baseline Analysis: Keyword Cloud**")
        all_keywords = " ".join(aspect_data['baseline_keywords'])
        if all_keywords.strip():
            try:
                wordcloud = WordCloud(width=400, height=300, background_color='white').generate(all_keywords)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation='bilinear');
                ax.axis('off')
                st.pyplot(fig)
            except Exception:
                st.text("Could not generate word cloud.")
        else:
            st.info("No evidence keywords found by Baseline.")

    with col2:
        st.markdown("**MetaMind Analysis: Mental States**")
        mental_state_data = aspect_data['review_top_hyp_type'].value_counts()
        if not mental_state_data.empty:
            st.bar_chart(mental_state_data)
            st.markdown(
                f"**Insight:** When users talk about **{selected_aspect}**, their dominant state is **{mental_state_data.index[0]}**.")
        else:
            st.info("No MetaMind data found.")


def display_anomaly_section(df_comparison: pd.DataFrame):
    """
    Displays the "Anomalies & Sarcasm Detector" section.
    """
    st.header("Anomaly & Sarcasm Detector")
    st.markdown("This table shows all reviews where the Baseline and MetaMind analyses disagreed...")

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
    gb.configure_column("Index", flex=0.5);
    gb.configure_column("Review", flex=3, wrapText=True, autoHeight=True)
    gb.configure_column("Baseline Sentiment", flex=1);
    gb.configure_column("MetaMind Sentiment", flex=1)
    grid_options = gb.build()
    AgGrid(df_anomalies, gridOptions=grid_options, height=400, width='100%', theme="streamlit",
           fit_columns_on_grid_load=True)


# === FIN DES AJOUTS ===


# --- Main Application (Fusionnée) ---

def main():
    st.set_page_config(page_title="MetaMind Sentiment Demo", layout="wide")
    st.title(" MetaMind Sentiment Demo")

    # === 1. Chargement des données pré-calculées (Pour les Dashboards) ===
    try:
        df_raw = load_deep_dive_data("data/clean/deep_dive_results.jsonl")
        df_processed = get_processed_data(df_raw)
        df_comparison = get_review_level_comparison_data(df_raw)

        # Affichage des Dashboards (AJOUTÉ)
        display_kpi_dashboard(df_comparison, df_processed)

    except Exception as e:
        st.error(f"Failed to load dashboard data: {e}")
        st.info("Live Analysis Only mode. Pre-computed dashboards are disabled.")
        # Initialiser des dataframes vides pour que l'app ne plante pas
        df_processed = pd.DataFrame()
        df_comparison = pd.DataFrame()

    # === 2. Section d'Analyse en Direct (Originale, mais modifiée) ===
    st.divider()
    st.header("Individual Analysis (Live Test)")
    st.markdown("Select any review from the table below to run a real-time analysis.")

    # --- MODIFICATION : Chargement du CSV pour ce tableau ---
    try:
        # load the csv of reviews from config
        csv_path = DATASET_CONFIG.get("reviews_csv_path")
        df_csv = pd.read_csv("data/clean/SamsungS25FE.csv", header=None, names=["Review"], sep="\t")
    except Exception as e:
        st.error(f"Could not load the reviews CSV file from path: {csv_path}")
        st.info(f"Error details: {e}")
        st.stop()
    # --- FIN DE LA MODIFICATION ---

    # display the tale (Utilise df_csv)
    gb = GridOptionsBuilder.from_dataframe(df_csv)
    gb.configure_selection("single", use_checkbox=False)
    grid_options = gb.build()

    grid_response = AgGrid(
        df_csv,  # <-- Modifié pour utiliser df_csv
        gridOptions=grid_options,
        height=250,
        theme="streamlit",
    )

    selected_rows = grid_response.get("selected_rows")
    if selected_rows is None:
        selected_rows = []

    if isinstance(selected_rows, pd.DataFrame):
        has_selection = not selected_rows.empty
    else:
        has_selection = len(selected_rows) > 0

    if has_selection:
        if isinstance(selected_rows, pd.DataFrame):
            selected_review = selected_rows.iloc[0]["Review"]
        else:
            selected_review = selected_rows[0]["Review"]

        if st.button("Analyze sentiment", type="primary"):
            llm = load_llm()
            context = []
            hypotheses = []  # <-- Gardé pour l'appel à 4 arguments

            with st.spinner("Running sentiment analysis..."):
                baseline_result = baseline_sentiment_json(llm, selected_review, context, max_retries=1)

                # --- Appel à 4 arguments (inchangé) ---
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

    # === 3. Affichage des autres dashboards (AJOUTÉ) ===
    st.divider()
    display_deep_dive_section(df_processed)
    st.divider()
    display_anomaly_section(df_comparison)


if __name__ == "__main__":
    main()