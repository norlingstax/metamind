from pathlib import Path
from typing import Tuple
import pandas as pd

from config import DATASET_CONFIG


def get_dataset_paths() -> Tuple[Path, Path]:
    """Return dataset CSV path and the derived deep-dive cache path."""
    dataset_path = Path(DATASET_CONFIG["reviews_csv_path"]).expanduser()
    deepdive_dir = Path(DATASET_CONFIG.get("deepdive_dir", dataset_path.parent)).expanduser()
    deepdive_suffix = DATASET_CONFIG.get("deepdive_suffix", "_deepdive.jsonl")
    deepdive_path = deepdive_dir / f"{dataset_path.stem}{deepdive_suffix}"
    return dataset_path, deepdive_path


def load_reviews_dataframe(dataset_path: Path) -> pd.DataFrame:
    """Load the reviews CSV as a single-column DataFrame."""
    separator = DATASET_CONFIG.get("reviews_csv_separator")
    try:
        if separator:
            return pd.read_csv(dataset_path, header=None, names=["Review"], sep=separator, engine="python")
        return pd.read_fwf(dataset_path, header=None, names=["Review"])
    except (pd.errors.ParserError, UnicodeDecodeError):
        return pd.read_fwf(dataset_path, header=None, names=["Review"])
