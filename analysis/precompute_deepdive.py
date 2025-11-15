from pathlib import Path
import json
import pandas as pd

from analysis.sentiment import metamind_sentiment_json
from baselines.raw_sentiment import baseline_sentiment_json
from config import LLM_CONFIG
from llm_interface.openai_llm import OpenAILLM
from utils.dataset_utils import get_dataset_paths, load_reviews_dataframe


def run_deepdive() -> Path:
    dataset_path, deepdive_path = get_dataset_paths()
    df = load_reviews_dataframe(dataset_path)
    if df.empty:
        raise RuntimeError(f"Dataset at {dataset_path} is empty.")

    llm = OpenAILLM(LLM_CONFIG)
    deepdive_path.parent.mkdir(parents=True, exist_ok=True)

    with deepdive_path.open("w", encoding="utf-8") as handle:
        for review in df["Review"].dropna():
            review_text = str(review).strip()
            if not review_text:
                continue
            baseline_res = baseline_sentiment_json(llm, review_text, [], max_retries=1)
            metamind_res = metamind_sentiment_json(llm, review_text, [], [])
            entry = {
                "review_text": review_text,
                "baseline_result": baseline_res,
                "metamind_result": metamind_res,
            }
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return deepdive_path


def main() -> None:
    deepdive_path = run_deepdive()
    print(f"Deep dive cache written to {deepdive_path}")


if __name__ == "__main__":
    main()
