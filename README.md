## Project Overview

This repository builds on top of the [MetaMind multi‑agent framework](https://github.com/XMZhangAI/MetaMind) and adapts it for product review analysis and marketing insight generation.

We only document our additions and how to run them. For the original MetaMind system (architecture, configuration details, and background), please refer to the upstream project.

## What We Added

- ToM‑informed sentiment pipeline
  - Auto‑generates Theory‑of‑Mind (ToM) hypotheses and applies Domain scoring (p_cond, p_prior, information gain, composite score) when hypotheses aren’t provided.
  - LLM‑based sentiment synthesis consolidates top hypotheses into a single sentiment JSON.
  - Aspect extraction labels product‑relevant dimensions (e.g., quality, UX, pricing) with evidence.
  - Recommendation stage summarizes drivers and suggests concise actions for neutral/negative cases.

- Baseline comparator
  - One‑shot, single‑prompt sentiment JSON (no ToM) for quick comparison.

- Streamlit demo UI
  - Side‑by‑side Baseline vs MetaMind outputs.
  - Dataset path is configurable in `config.py`.

- Robustness
  - Strict JSON prompts + guarded parsing and small retries.
  - Optional, config‑driven fallback used only when JSON is malformed or incomplete.

- Deep-dive dashboard & cache
  - Optional toggle (`DATASET_CONFIG["deepdive_enabled"]`) to show aggregate charts.
  - Cached JSONL per dataset (`<dataset>_deepdive.jsonl` under `data/processed/`) generated via `python -m analysis.precompute_deepdive`.
  - When the toggle is on but the JSONL cache is missing, the UI stays in the lightweight mode and shows a reminder to compute the cache first.

For a deeper, code‑level pipeline description, see [DOCUMENTATION.md](https://github.com/norlingstax/metamind/blob/main/DOCUMENTATION.md).

## Why It Matters (Marketing Impact)

- Aspect‑level sentiment pinpoints praise/pain points with short evidence.
- ToM hypotheses improve explainability (beliefs, desires, intentions, emotions behind sentiment).
- Recommendations turn insights into next steps for product and CX teams.
- Baseline vs MetaMind makes value measurable; trend tracking over releases is straightforward.

## Data Source

The demo datasets are scraped from GSMArena review comments ([Iphone](https://www.gsmarena.com/reviewcomm-2886.php), [Samsung](https://www.gsmarena.com/reviewcomm-2880p2.php)).

## How to Run

1) Environment

Create a virtual environment and activate it:

``` bash
python -m venv .venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

2) Install dependencies

``` bash
pip install -r requirements.txt
```

3) Configure

- Open `config.py` and set `LLM_CONFIG` (model, base_url, api key as appropriate).
- Set `DATASET_CONFIG["reviews_csv_path"]` to your CSV of reviews.
- Place your CSV under `data/raw/` (default) so caches are stored under `data/processed/` with names like `Iphone17_deepdive.jsonl`.
- Turn `DATASET_CONFIG["deepdive_enabled"]` on/off to control the aggregate dashboards. When it's on, run `python -m analysis.precompute_deepdive` once per dataset to populate the cache.
- For deeper MetaMind configs/architecture, see the upstream repo.

4) Run the Streamlit demo

``` bash
python -m streamlit run interface.py
```

In the app:
- Select a review from the table
- Click "Analyze sentiment" to compare Baseline (single‑pass) vs MetaMind (ToM → Domain → synthesis → aspects → recommendation)
