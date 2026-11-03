from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

USER_COL = "UserId"
ITEM_COL = "ProductId"
RATING_COL = "Rating"
TIME_COL = "Timestamp"


def to_implicit(df: pd.DataFrame, rating_threshold: float) -> pd.DataFrame:
    """Keep only interactions whose rating is treated as a positive signal."""
    out = df.loc[df[RATING_COL] >= rating_threshold, [USER_COL, ITEM_COL, TIME_COL]]
    return out.drop_duplicates(subset=[USER_COL, ITEM_COL])


def temporal_split(
    df: pd.DataFrame, val_frac: float, test_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split interactions into train/val/test by global timestamp cutoffs (no leakage)."""
    train_frac = 1.0 - val_frac - test_frac
    if train_frac <= 0:
        raise ValueError("val_frac + test_frac must be < 1")

    timestamps = df[TIME_COL].sort_values()
    train_cutoff = timestamps.quantile(train_frac)
    val_cutoff = timestamps.quantile(train_frac + val_frac)

    train = df[df[TIME_COL] <= train_cutoff]
    val = df[(df[TIME_COL] > train_cutoff) & (df[TIME_COL] <= val_cutoff)]
    test = df[df[TIME_COL] > val_cutoff]
    return train, val, test


def k_core_filter(df: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    """Iteratively drop users/items below the interaction threshold until stable."""
    current = df
    while True:
        user_counts = current[USER_COL].value_counts()
        item_counts = current[ITEM_COL].value_counts()
        keep_users = user_counts[user_counts >= min_user].index
        keep_items = item_counts[item_counts >= min_item].index
        filtered = current[current[USER_COL].isin(keep_users) & current[ITEM_COL].isin(keep_items)]
        if len(filtered) == len(current):
            return filtered
        current = filtered


@dataclass
class Vocab:
    ids: list
    index: dict

    @classmethod
    def from_values(cls, values) -> "Vocab":
        ids = sorted(set(values))
        return cls(ids=ids, index={v: i for i, v in enumerate(ids)})

    def __len__(self) -> int:
        return len(self.ids)


def split_cold_items(
    df: pd.DataFrame, user_vocab: Vocab, item_vocab: Vocab
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict to users known from train; separate rows into warm vs cold-item slices."""
    known_users = df[df[USER_COL].isin(user_vocab.index)]
    warm = known_users[known_users[ITEM_COL].isin(item_vocab.index)]
    cold_items = known_users[~known_users[ITEM_COL].isin(item_vocab.index)]
    return warm, cold_items


def build_interaction_matrix(
    df: pd.DataFrame, user_vocab: Vocab, item_vocab: Vocab
) -> sparse.csr_matrix:
    """Build a binary (users x items) implicit interaction matrix for the given vocab."""
    rows = df[USER_COL].map(user_vocab.index).to_numpy()
    cols = df[ITEM_COL].map(item_vocab.index).to_numpy()
    data = np.ones(len(df), dtype="float32")
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(user_vocab), len(item_vocab)))
