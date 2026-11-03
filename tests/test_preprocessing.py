import pandas as pd

from recsys.preprocessing import (
    ITEM_COL,
    USER_COL,
    Vocab,
    k_core_filter,
    split_cold_items,
    temporal_split,
    to_implicit,
)


def _toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            USER_COL: ["u1", "u1", "u2", "u2", "u3"],
            ITEM_COL: ["i1", "i2", "i1", "i3", "i4"],
            "Rating": [5.0, 3.0, 4.0, 5.0, 2.0],
            "Timestamp": [100, 200, 300, 400, 500],
        }
    )


def test_to_implicit_filters_by_threshold():
    df = _toy_df()
    out = to_implicit(df, rating_threshold=4)
    # rows with Rating >= 4: (u1,i1,5.0), (u2,i1,4.0), (u2,i3,5.0)
    assert len(out) == 3
    assert "Rating" not in out.columns
    pairs = set(zip(out[USER_COL], out[ITEM_COL], strict=True))
    assert pairs == {("u1", "i1"), ("u2", "i1"), ("u2", "i3")}


def test_temporal_split_respects_time_ordering():
    df = _toy_df()
    train, val, test = temporal_split(df, val_frac=0.2, test_frac=0.2)
    if len(train) and len(val):
        assert train["Timestamp"].max() <= val["Timestamp"].min()
    if len(val) and len(test):
        assert val["Timestamp"].max() <= test["Timestamp"].min()
    assert len(train) + len(val) + len(test) == len(df)


def test_k_core_filter_enforces_thresholds():
    df = pd.DataFrame(
        {
            USER_COL: ["u1", "u1", "u1", "u2", "u3"],
            ITEM_COL: ["i1", "i2", "i3", "i1", "i4"],
            "Timestamp": [1, 2, 3, 4, 5],
        }
    )
    filtered = k_core_filter(df, min_user=2, min_item=1)
    # u2 and u3 only have 1 interaction each -> dropped; u1 (3 interactions) kept.
    assert set(filtered[USER_COL]) == {"u1"}


def test_k_core_filter_converges_iteratively():
    # u2's only item (i1) is shared with u1, but after dropping u3 (1 interaction),
    # i4 disappears too; min_item=2 should also drop i1 since only u1 keeps it alone.
    df = pd.DataFrame(
        {
            USER_COL: ["u1", "u1", "u2"],
            ITEM_COL: ["i1", "i2", "i1"],
            "Timestamp": [1, 2, 3],
        }
    )
    filtered = k_core_filter(df, min_user=2, min_item=2)
    assert len(filtered) == 0


def test_split_cold_items_buckets_unknown_items():
    user_vocab = Vocab.from_values(["u1", "u2"])
    item_vocab = Vocab.from_values(["i1", "i2"])
    df = pd.DataFrame(
        {
            USER_COL: ["u1", "u1", "u3"],
            ITEM_COL: ["i1", "i_unseen", "i1"],
            "Timestamp": [1, 2, 3],
        }
    )
    warm, cold = split_cold_items(df, user_vocab, item_vocab)
    # u3 is an unknown user -> dropped entirely.
    assert len(warm) + len(cold) == 2
    assert list(warm[ITEM_COL]) == ["i1"]
    assert list(cold[ITEM_COL]) == ["i_unseen"]
