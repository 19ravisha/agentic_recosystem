from pathlib import Path

import pandas as pd


def load_ratings(csv_path: str | Path) -> pd.DataFrame:
    """Load the Amazon Beauty ratings CSV (UserId, ProductId, Rating, Timestamp)."""
    df = pd.read_csv(
        csv_path,
        dtype={"UserId": "category", "ProductId": "category", "Rating": "float32"},
    )
    df["Timestamp"] = df["Timestamp"].astype("int64")
    return df
