import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.config import load_config
from recsys.data import load_ratings
from recsys.evaluate import evaluate_model
from recsys.models import PopularityRecommender, SVDRecommender
from recsys.preprocessing import (
    ITEM_COL,
    USER_COL,
    Vocab,
    build_interaction_matrix,
    k_core_filter,
    split_cold_items,
    temporal_split,
    to_implicit,
)
from recsys.utils import set_seed


def _print_metrics(name: str, results: dict, k_list: list[int]) -> None:
    for k in k_list:
        print(f"  {name} Recall@{k}={results[k]['recall']:.4f} NDCG@{k}={results[k]['ndcg']:.4f}")


def run_pipeline(config_path: str) -> None:
    config = load_config(config_path)
    set_seed(config["seed"])

    print(f"Loading ratings from {config['paths']['raw']}")
    raw = load_ratings(config["paths"]["raw"])
    print(f"  {len(raw):,} raw rows")

    implicit = to_implicit(raw, config["data"]["positive_rating_threshold"])
    print(f"  {len(implicit):,} positive (implicit) interactions")

    train_raw, val_raw, test_raw = temporal_split(
        implicit, config["split"]["val_frac"], config["split"]["test_frac"]
    )

    train = k_core_filter(
        train_raw,
        config["data"]["min_user_interactions"],
        config["data"]["min_item_interactions"],
    )
    print(
        f"  train: {len(train_raw):,} -> {len(train):,} rows after k-core filtering "
        f"({train[USER_COL].nunique():,} users, {train[ITEM_COL].nunique():,} items)"
    )

    user_vocab = Vocab.from_values(train[USER_COL])
    item_vocab = Vocab.from_values(train[ITEM_COL])

    val_warm, val_cold = split_cold_items(val_raw, user_vocab, item_vocab)
    test_warm, test_cold = split_cold_items(test_raw, user_vocab, item_vocab)
    print(f"  val: {len(val_warm):,} warm / {len(val_cold):,} cold-item rows")
    print(f"  test: {len(test_warm):,} warm / {len(test_cold):,} cold-item rows")

    train_matrix = build_interaction_matrix(train, user_vocab, item_vocab)
    val_matrix = build_interaction_matrix(val_warm, user_vocab, item_vocab)
    test_matrix = build_interaction_matrix(test_warm, user_vocab, item_vocab)

    # Cold items are by definition absent from item_vocab, so Popularity/SVD (no
    # content features) can never recommend them -- Recall/NDCG against this
    # slice would always be empty. Report the volume as a visibility metric
    # instead (see docs/decisions.md D1/D2).
    test_total = len(test_warm) + len(test_cold)
    cold_frac = len(test_cold) / test_total if test_total else 0.0
    print(f"  cold-item share of test interactions: {cold_frac:.1%} (unrecoverable by this model)")

    n_users = len(user_vocab)
    max_components = min(train_matrix.shape) - 1
    tuning_k = config["eval"]["tuning_k"]

    best_n_components, best_recall = config["model"]["svd"]["default_n_components"], -1.0
    for n_components in config["model"]["svd"]["n_components_grid"]:
        n_components = min(n_components, max_components)
        candidate = SVDRecommender(n_components=n_components, random_state=config["seed"])
        candidate.fit(train_matrix)
        val_results = evaluate_model(candidate, n_users, train_matrix, val_matrix, [tuning_k])
        recall = val_results[tuning_k]["recall"]
        print(f"  SVD n_components={n_components}: val Recall@{tuning_k}={recall:.4f}")
        if recall > best_recall:
            best_recall, best_n_components = recall, n_components

    popularity = PopularityRecommender()
    popularity.fit(train_matrix)

    svd = SVDRecommender(n_components=best_n_components, random_state=config["seed"])
    svd.fit(train_matrix)

    k_list = config["eval"]["k_list"]
    print("\nTest metrics (warm items):")
    for name, model in [("Popularity", popularity), ("SVD", svd)]:
        _print_metrics(
            name, evaluate_model(model, n_users, train_matrix, test_matrix, k_list), k_list
        )

    top_n = config["output"]["top_n"]
    top_k_idx, top_k_scores = svd.recommend(n_users, top_n, train_matrix)

    user_ids = np.array(user_vocab.ids, dtype=object)
    item_ids = np.array(item_vocab.ids, dtype=object)
    rec_df = pd.DataFrame(
        {
            "user_id": np.repeat(user_ids, top_n),
            "rank": np.tile(np.arange(1, top_n + 1), n_users),
            "product_id": item_ids[top_k_idx.ravel()],
            "score": top_k_scores.ravel(),
        }
    )

    out_path = Path(config["paths"]["recommendations"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rec_df.to_parquet(out_path, index=False)
    print(f"\nWrote {len(rec_df):,} recommendations for {n_users:,} users to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the recsys training/eval pipeline")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
